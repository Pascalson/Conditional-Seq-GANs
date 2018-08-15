import tensorflow as tf
import numpy as np
import copy

import data_utils
from units import *
from critic import *

def load_critic(name, size=None, num_layers=None, vocab_size=None, buckets=None):
    print(name)
    if name == None:
        return None
    with variable_scope.variable_scope('critic') as scope:
        if name == 'Counting_Task':
            return Counting_Task()
        elif name == 'Sequence_Task':
            return Sequence_Task()
        elif name == 'Addition_Task':
            return Addition_Task()
        elif 'SeqGAN' in name or 'MaliGAN' in name:
            return StepGAN(size, num_layers, vocab_size, buckets)
        elif name == 'REGS':
            return StepGAN(size, num_layers, vocab_size, buckets)
        elif name == 'MaskGAN':
            return StepGAN(size, num_layers, vocab_size, buckets)
        elif 'StepGAN' in name:
            return StepGAN(size, num_layers, vocab_size, buckets)
        #elif name == 'Direct_WGAN_GP':
        #    return Direct_WGAN_GP(size, num_layers, vocab_size, buckets)

class Seq2Seq:#SeqGenerator
    def __init__(
            self,
            mode,
            size,
            num_layers,
            vocab_size,
            buckets,
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            max_gradient_norm=5.0,
            critic=None,
            critic_size=None,
            critic_num_layers=None,
            other_option=None,
            use_attn=False,
            output_sample=False,
            input_embed=True,
            feed_prev=False,
            batch_size=32,
            D_lr=1e-4,
            D_lr_decay_factor=0.5,
            v_lr=1e-4,
            v_lr_decay_factor=0.5,
            dtype=tf.float32):

        self.train_sample_loop_coe = 1
        self.test_sample_loop_coe = 1
        self.train_Monte_Carlo_N = 5
        # self-config
        self.size = size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        if vocab_size > 1000:
            num_sampled = 512
        elif vocab_size < 20:
            num_sampled = 5
        self.buckets = buckets
        self.other_option = other_option
        self.use_attn = use_attn# has been decrepted
        self.output_sample = output_sample
        self.input_embed = input_embed
        self.feed_prev = feed_prev
        self.batch_size = batch_size
        # general vars: learning rate, global steps
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.D_lr = tf.Variable(float(D_lr), trainable=False, dtype=dtype)
        self.v_lr = tf.Variable(float(v_lr), trainable=False, dtype=dtype)
        self.op_lr_decay = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.op_D_lr_decay = self.D_lr.assign(self.D_lr * D_lr_decay_factor)
        self.op_v_lr_decay = self.v_lr.assign(self.v_lr * v_lr_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.global_D_step = tf.Variable(0, trainable=False)
        self.global_V_step = tf.Variable(0, trainable=False)
        # building critics or discriminators
        self.critic = load_critic(critic, critic_size, critic_num_layers, vocab_size, buckets)
        self.critic_name = critic
        # building value network
        if critic is not None and critic is not 'None':
            with variable_scope.variable_scope('valuenet') as scope:
                self.value_net = ValueNet(critic_size, critic_num_layers, vocab_size, buckets)
        
        # core cells, encoder and decoder are separated
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)

        # output projection
        w = tf.get_variable('proj_w', [size, vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable('proj_b', [vocab_size])
        self.output_projection = (w, b)
        # input embedding
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])

        # seq2seq-specific functions
        def loop_function(prev):
            # used in decoder, feeding the previous argmax output as the next input
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_symbol = math_ops.argmax(prev, axis=1)
            emb_prev = embedding_ops.embedding_lookup(self.embedding, prev_symbol)
            return emb_prev
        
        def sample_loop_function(prev):
            # used in decoder, feeding the previous output
            # sampled from softmax distribution as the next input
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_index = tf.multinomial(tf.log(tf.nn.softmax(self.train_sample_loop_coe*prev)), 1)
            prev_symbol = tf.reshape(prev_index, [-1])
            emb_prev = embedding_ops.embedding_lookup(self.embedding, prev_symbol)
            return [emb_prev, prev_symbol]

        def test_sample_loop_function(prev):
            # sample_loop_function() used in testing stage,
            # by this, we can test the influence of sample_loop_coe(coefficient)
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_index = tf.multinomial(tf.log(tf.nn.softmax(self.test_sample_loop_coe*prev)), 1)
            prev_symbol = tf.reshape(prev_index, [-1])
            emb_prev = embedding_ops.embedding_lookup(self.embedding, prev_symbol)
            return [emb_prev, prev_symbol]

        def softmax_loss_function(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(tf.nn.sampled_softmax_loss(
                weights = local_w_t,
                biases = local_b,
                inputs = local_inputs,
                labels = labels,
                num_sampled = num_sampled,
                num_classes = vocab_size),
                dtype = tf.float32)

        def compute_loss(logits, targets, weights):
            with ops.name_scope("sequence_loss", logits + targets + weights):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                cost = math_ops.reduce_sum(log_perps)
                batch_size = array_ops.shape(targets[0])[0]
                return cost / math_ops.cast(batch_size, cost.dtype)

        def get_eos_value(rewards, uniform_weights):
            # used in decoder, getting the value at the first generated <EOS> token
            eos = [tf.cast(tf.equal(math_ops.add_n(uniform_weights[:i+1]), math_ops.add_n(uniform_weights)), tf.float32)*uniform_weights[i] for i in range(len(rewards))]
            outs = []
            for r, w in zip(rewards, eos):
                outs.append(tf.reshape(r, [-1]) * w)
            eos_value = math_ops.add_n(outs)
            return eos_value

        def uniform_weights(targets):
            # used inj decoder, generating uniform weights at all time steps
            # until the first generated <EOS> token
            tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
            tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
            uniform_weights = \
                    [tf.cast(tf.equal(math_ops.add_n(tmp[i:]), math_ops.add_n(tmp)), tf.float32) \
                     for i in range(len(tmp))]
            return uniform_weights

        def weighted_rewards(rewards, targets, uniform_weights, method='uniform'):
            if method == 'uniform':
                weights = uniform_weights
            elif method == 'random':#FIXME
                rand = tf.random_uniform([1],maxval=tf.cast(len(uniform_weights),tf.int32),dtype=tf.int32)
                weights = []
                for i in range(len(uniform_weights)):
                    weights.append(tf.cond(tf.equal(i,tf.reshape(rand,[])),
                                           lambda: tf.ones(tf.shape(targets[0])),
                                           lambda: tf.zeros(tf.shape(targets[0]))))
            elif method == 'decrease':
                weights = [math_ops.add_n(uniform_weights[i:]) for i in range(len(uniform_weights))]
            elif method == 'increase':
                weights = [math_ops.add_n(uniform_weights[:(i+1)]) * uniform_weights[i] for i in range(len(uniform_weights))]
            outs = []
            for r, w in zip(rewards, weights):
                outs.append(tf.reshape(r, [-1]) * w)
            return outs, math_ops.add_n(weights), math_ops.add_n(uniform_weights)

        def seq_log_prob(logits, targets, rewards=None):
            if rewards is None:
                rewards = [tf.ones(tf.shape(target), tf.float32) for target in targets]
            with ops.name_scope("sequence_log_prob", logits + targets + rewards):
                log_perp_list = []
                tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
                tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
                weights = [math_ops.add_n(tmp[i:]) for i in range(len(tmp))]
                for logit, target, weight, reward in zip(logits, targets, weights, rewards):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight * reward)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                return log_perps

        def each_perp(logits, targets, weights):
            with ops.name_scope("each_perp", logits + targets):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                return log_perp_list 

        # encoder's placeholder
        self.encoder_inputs = []
        for bid in range(buckets[-1][0]):
            self.encoder_inputs.append(
                tf.placeholder(tf.int32, shape = [None],
                               name = 'encoder{0}'.format(bid)))
        self.seq_len = tf.placeholder(
            tf.int32, shape = [None],
            name = 'enc_seq_len')

        # decoder's placeholder
        self.decoder_inputs = []
        self.target_weights = []
        if not feed_prev and mode == 'TRAIN':
            for bid in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(
                    tf.placeholder(tf.int32, shape = [None],
                                   name = 'decoder{0}'.format(bid)))
                self.target_weights.append(
                    tf.placeholder(tf.float32, shape = [None],
                                   name = 'weight{0}'.format(bid)))
            targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs)-1)]
        elif mode == 'TEST' or mode == 'D_TEST':
            for bid in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(
                    tf.placeholder(tf.int32, shape = [None],
                                   name = 'decoder{0}'.format(bid)))
        else:
            self.decoder_inputs = [tf.placeholder(tf.int32, shape = [None], name = 'decoder0')]
        # other placeholders
        if critic is not None:
            self.fed_samples = [ tf.placeholder(tf.int32, shape = [None], name = 'fed_sample{0}'.format(i)) for i in range(buckets[-1][1]) ]
            self.fed_rewards = tf.placeholder(tf.float32, shape = [None], name = 'fed_rewards')

        # the operations of this sequence generator: training, testing
        if mode == 'TRAIN':
            # critic None: training by Maximum-Likelihood Estimation (MLE)
            # others: REINFORCE (policy gradient) based methods,
            # include REINFORCE, SeqGAN, MaliGAN, REGS, ESGAN

            # for debug
            self.debug1 = []
            # for generating
            self.enc_state = []# shared with MLE
            self.outputs = []# shared with MLE
            self.samples_dists = []
            self.each_probs = []
            self.perp = []
            # for generating by REINFORCE
            self.out_dists = []

            # for rewarding of policy gradient
            self.each_rewards = []
            self.rewards = []
            self.for_G_rewards = []

            # for training
            self.losses = []# shared with MLE
            self.value_losses = []
            self.D_losses = []
            self.D_real = []
            self.D_fake = []

            def Monte_Carlo():
                # Monte-Carlo tree search
                N = self.train_Monte_Carlo_N
                rewards = []
                for step in range(bucket[1]):
                    each_step_reward = []
                    for _ in range(N):
                        with variable_scope.variable_scope(
                            variable_scope.get_variable_scope(), reuse=True):
                            _, MC_sample, _ = \
                                decode(self.cell, hiddens[step], self.embedding, \
                                       [samples[step]], bucket[1]-step-1, \
                                       feed_prev=True, loop_function=sample_loop_function)
                            each_prob_fake, _ = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples[:step]+MC_sample, batch_size)
                            fake_uniW = uniform_weights(samples[:step]+MC_sample)
                            r = get_eos_value(each_prob_fake, fake_uniW)
                        each_step_reward.append(tf.reshape(r,[-1]))
                    rewards.append(math_ops.add_n(each_step_reward) / N)
                return rewards

            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                    enc_outputs, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)
                    if critic is None:
                        outputs, _, _ = \
                            decode(self.cell, enc_state, self.embedding, \
                                   self.decoder_inputs[:bucket[1]], \
                                   bucket[1]+1, feed_prev=False)
                    else:
                        samples_dists, samples, hiddens = \
                            decode(self.cell, enc_state, self.embedding, \
                                   [self.decoder_inputs[0]], bucket[1], \
                                   feed_prev=True, loop_function=sample_loop_function)
                        prob = - seq_log_prob(samples_dists, samples)
                        
                    if critic is None:
                        loss = compute_loss(outputs, targets[:bucket[1]], \
                                            self.target_weights[:bucket[1]])
                    elif 'GAN' in critic or critic == 'REGS':
                        # building discriminator and value network
                        each_prob_fake, each_logit_fake = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                        each_prob_fake_value, each_logit_fake_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                        with variable_scope.variable_scope(
                                variable_scope.get_variable_scope(), reuse=True):
                            each_prob_real, each_logit_real = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)
                            each_prob_real_value, each_logit_real_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)

                        # uniform weights
                        fake_uniW = uniform_weights(samples)
                        real_uniW = uniform_weights(self.critic.real_data[:bucket[1]])

                        if 'SeqGAN' in critic or 'MaliGAN' in critic:
                            # end of sentence
                            for_D_score_fake = get_eos_value(each_prob_fake, fake_uniW)
                            for_D_score_real = get_eos_value(each_prob_real, real_uniW)

                        elif critic == 'REGS':
                            # get random subsequence
                            for_D_each_prob_fake, for_D_fake_credits, _ = \
                                weighted_rewards(each_prob_fake, samples, fake_uniW, 'random')
                            for_D_each_prob_real, for_D_real_credits, _ = \
                                weighted_rewards(each_prob_real, self.critic.real_data[:bucket[1]], \
                                                 real_uniW, 'random')
                            for_D_score_fake = math_ops.add_n(for_D_each_prob_fake)
                            for_D_score_real = math_ops.add_n(for_D_each_prob_real)

                        elif 'StepGAN' in critic or 'MaskGAN' in critic:
                            # get uniform all scores of D
                            for_D_each_prob_fake, for_D_fake_credits, _ = \
                                weighted_rewards(each_prob_fake, samples, fake_uniW, 'uniform')
                            for_D_each_prob_real, for_D_real_credits, _ = \
                                weighted_rewards(each_prob_real, self.critic.real_data[:bucket[1]], \
                                                 real_uniW, 'uniform')
                            if 'StepGAN' in critic:
                                for_D_score_fake = math_ops.add_n(for_D_each_prob_fake) / (for_D_fake_credits + 1e-12)
                                for_D_score_real = math_ops.add_n(for_D_each_prob_real) / (for_D_real_credits + 1e-12)
                            else:
                                #for_D_score_fake = get_eos_value(each_prob_fake, fake_uniW)
                                #for_D_score_real = get_eos_value(each_prob_real, real_uniW)
                                for_D_score_fake = math_ops.add_n([tf.log(1. - each_prob+1e-12)*uniW for each_prob, uniW in zip(each_prob_fake, fake_uniW)]) / (for_D_fake_credits + 1e-12)
                                for_D_score_real = math_ops.add_n([tf.log(each_prob+1e-12)*uniW for each_prob, uniW in zip(each_prob_real, real_uniW)]) / (for_D_real_credits + 1e-12)

                        if 'MaskGAN' in critic:
                            D_loss = -tf.reduce_mean(for_D_score_real + for_D_score_fake)
                        else:
                            # training D
                            D_loss = -tf.reduce_mean(tf.log(for_D_score_real) + tf.log(1.-for_D_score_fake))
                                

                        # print reward
                        if 'SeqGAN' in critic or 'MaliGAN' in critic:
                            reward = tf.reshape(for_D_score_fake, [-1])# FIXME
                            D_prob_fake = for_D_score_fake
                            D_prob_real = for_D_score_real
                        else:
                            reward = tf.reduce_mean(for_D_score_fake)
                            D_prob_fake = [[r[b] for r in for_D_each_prob_fake] for b in range(batch_size)]
                            D_prob_real = [[r[b] for r in for_D_each_prob_real] for b in range(batch_size)]

                        if 'SeqGAN' in critic:
                            returns = [D_prob_fake for i in range(bucket[1])]
                            real_returns = [D_prob_real for i in range(bucket[1])]
                        elif 'MaliGAN' in critic:# FIXME
                            returns = [D_prob_fake/tf.reduce_sum(D_prob_fake) for i in range(bucket[1])]
                            real_returns = [D_prob_real/tf.reduce_sum(D_prob_real) for i in range(bucket[1])]
                        elif critic == 'REGS':
                            returns = [each_prob_fake[i]*fake_uniW[i] for i in range(bucket[1])]
                            real_returns = [each_prob_real[i]*real_uniW[i] for i in range(bucket[1])]
                        elif 'StepGAN' in critic or critic == 'MaskGAN':
                            uni_each_prob_fake = for_D_each_prob_fake
                            uni_each_prob_real = for_D_each_prob_real
                            if 'seq' in critic or critic == 'MaskGAN':
                                returns = [math_ops.add_n(uni_each_prob_fake[i:]) for i in range(bucket[1])]
                                real_returns = [math_ops.add_n(uni_each_prob_real[i:]) for i in range(bucket[1])]
                            else:
                                returns = uni_each_prob_fake
                                real_returns = uni_each_prob_real

                        if 'MC' in critic:
                            # Monte Carlo
                            returns = Monte_Carlo()

                        # FIXME REGS for_G_rewards = [tf.reshape((each_prob_fake[i] - each_prob_fake_value[i]),[-1])*fake_uniW[i] for i in range(bucket[1])]
                        if critic == 'REGS':
                            for_G_rewards = [tf.reshape((each_prob_fake[i] - each_prob_fake_value[i]),[-1])*fake_uniW[i] for i in range(bucket[1])]
                        else: 
                            minus_baseline = [returns[i] - tf.reshape(each_prob_fake_value[i],[-1]) for i in range(bucket[1])]
                            if '-W' in critic:
                                for_G_prob_fake, for_G_fake_credits, _ = \
                                    weighted_rewards(minus_baseline, samples, fake_uniW, 'decrease')
                            else:
                                for_G_prob_fake, for_G_fake_credits, _ = \
                                    weighted_rewards(minus_baseline, samples, fake_uniW, 'uniform')
                            for_G_rewards = for_G_prob_fake

                        value_loss_update = tf.reduce_sum([tf.square(returns[i] - tf.reshape(each_prob_fake_value[i],[-1]))*fake_uniW[i] for i in range(bucket[1])]) / (tf.reduce_sum(fake_uniW)+1e-12)
                        value_loss_update += tf.reduce_sum([tf.square(real_returns[i] - tf.reshape(each_prob_real_value[i],[-1]))*real_uniW[i] for i in range(bucket[1])]) / (tf.reduce_sum(real_uniW)+1e-12)
                        loss_update = each_perp(samples_dists, samples, fake_uniW)

                    #REINFORCE
                    else:
                        with variable_scope.variable_scope(
                            variable_scope.get_variable_scope(), reuse=True):
                            out_dist, hiddens, _ = \
                                decode(self.cell, enc_state, self.embedding, \
                                       self.decoder_inputs[:bucket[1]], bucket[1], \
                                       feed_prev=False)
                        self.out_dists.append(out_dist)
                        reward = self.fed_rewards
                        loss = seq_log_prob(out_dist, self.fed_samples[:bucket[1]], \
                                            [reward - tf.reduce_mean(reward) for _ in range(bucket[1])])
                        loss_update = tf.reduce_sum(loss) / batch_size

                    if critic is None:
                        self.enc_state.append(enc_state)# useless
                        self.outputs.append(outputs)# useless, for print
                        self.losses.append(loss)
                        
                    else:
                        # for debug
                        self.debug1.append(for_D_score_fake)
                        # for generating
                        self.enc_state.append(enc_state)# useless
                        self.outputs.append(samples)# useless, for print
                        self.samples_dists.append(samples_dists)# useless
                        self.each_probs.append(prob)# useless, for print
                        self.perp.append(tf.reduce_sum(prob) / batch_size)# useless, for print
                        
                        # for scoring, REINFORCE's reward is got from environment
                        self.each_rewards.append(reward)# useless
                        self.for_G_rewards.append(for_G_rewards)

                        # for training
                        self.losses.append(loss_update)
                        self.value_losses.append(value_loss_update)
                        if 'GAN' in critic or critic == 'REGS':# for print
                            self.D_losses.append(D_loss)
                            self.D_real.append(D_prob_real)
                            self.D_fake.append(D_prob_fake)

            # all parameters collection
            params = tf.trainable_variables()
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='valuenet')
            s2s_params = [ x for x in params if x not in critic_params and x not in value_params ]
            critic_params.append(self.global_D_step)
            value_params.append(self.global_V_step)
            # TODO print(s2s_params)
            # TODO print(critic_params)

            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            D_optimizer = tf.train.GradientDescentOptimizer(self.D_lr)

            # update operation
            self.op_update = []
            self.D_solver = []
            self.v_solver = []
            for j in range(len(self.buckets)):
                # update generator
                if critic is None:
                    gradients = tf.gradients(self.losses[j], s2s_params)
                else:
                    gradients = tf.gradients(self.losses[j], s2s_params, self.for_G_rewards[j])
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.op_update.append(optimizer.apply_gradients(
                    zip(clipped_gradients, s2s_params),
                    global_step=self.global_step))
                # update discriminator
                if critic is not None:
                    D_grads = tf.gradients(self.D_losses[j], critic_params)
                    clipped_D_grads, _ = tf.clip_by_global_norm(D_grads, max_gradient_norm)
                    self.D_solver.append(D_optimizer.apply_gradients(
                        zip(clipped_D_grads, critic_params),
                        global_step=self.global_D_step))
                    # update value net
                    v_grads = tf.gradients(self.value_losses[j], value_params)
                    clipped_v_grads, _ = tf.clip_by_global_norm(v_grads, max_gradient_norm)
                    self.v_solver.append(D_optimizer.apply_gradients(
                        zip(clipped_v_grads, value_params),
                        global_step=self.global_V_step))

            self.pre_D_saver = tf.train.Saver(var_list=critic_params, max_to_keep=None, sharded=True)
            self.pre_value_saver = tf.train.Saver(var_list=value_params, max_to_keep=None, sharded=True)
            self.pre_saver = tf.train.Saver(var_list=s2s_params, sharded=True)

        elif mode == 'TEST':
            self.enc_state = []
            self.outputs = []
            enc_outputs, enc_state = \
                encode(self.enc_cell, self.encoder_inputs, self.seq_len)
            
            # for beam search, probs
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=None):
                outputs, _, _ = \
                    decode(self.cell, enc_state, self.embedding, \
                           self.decoder_inputs, buckets[-1][1], \
                           feed_prev=False)
            self.outs = []
            for l in range(len(outputs)):
                outs = nn_ops.xw_plus_b(outputs[l], self.output_projection[0], self.output_projection[1])
                # axis used for probs, default dim=-1
                self.outs.append(tf.nn.softmax(outs))

            
            # for MMI
            local_batch_size = array_ops.shape(self.decoder_inputs[0])[0]
            lm_outputs, _, _ = \
                decode(self.cell, self.cell.zero_state(local_batch_size, tf.float32), self.embedding, \
                       self.decoder_inputs, buckets[-1][1], \
                       feed_prev=False)
            self.lm_outs = []
            for l in range(len(lm_outputs)):
                lm_outs = nn_ops.xw_plus_b(lm_outputs[l], self.output_projection[0], self.output_projection[1])
                self.lm_outs.append(tf.nn.softmax(lm_outs))
            
            self.enc_state.append(enc_state)

            # for argmax test
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=None):
                outputs, _, _ = \
                    decode(self.cell, enc_state, self.embedding, \
                           self.decoder_inputs, buckets[-1][1], \
                           feed_prev=True, loop_function=loop_function)
            self.outputs.append(outputs)
            self.print_outputs = []
            self.tmp_outputs = []
            self.prob_outputs = []
            for j, outs in enumerate(self.outputs):
                self.print_outputs.append([])
                self.tmp_outputs.append([])
                self.prob_outputs.append([])
                for i in range(len(outs)):
                    self.print_outputs[j].append(nn_ops.xw_plus_b(outs[i], self.output_projection[0], self.output_projection[1]))
                    self.tmp_outputs[j].append(math_ops.argmax(self.print_outputs[j][i], axis=1))
                    self.prob_outputs[j].append(tf.nn.softmax(self.print_outputs[j][i]))

            self.max_log_prob = - seq_log_prob(outputs, self.tmp_outputs[0])

            # for sample test
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True):
                tmp, self.samples, _ = \
                    decode(self.cell, enc_state, self.embedding, \
                           self.decoder_inputs, buckets[-1][1], \
                           feed_prev=True, loop_function=test_sample_loop_function)
            self.log_prob = - seq_log_prob(tmp, self.samples)
            params = tf.trainable_variables()
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            s2s_params = [ x for x in params if x not in critic_params ]
            general_s2s_params = {}
            other_ver = False

            self.pre_saver = tf.train.Saver(var_list=s2s_params, sharded=True)

        elif mode == 'D_TEST':
            print(self.critic_name)
            print('discriminator testing  mode (D_TEST)')
            each_prob, each_logit = self.critic.discriminator(self.encoder_inputs, self.seq_len, self.critic.real_data, batch_size)
            each_value_prob, each_value_logit = self.value_net.discriminator(self.encoder_inputs, self.seq_len, self.critic.real_data, batch_size)
            real_uniW = uniform_weights(self.critic.real_data)
            if 'SeqGAN' in self.critic_name or 'MaliGAN' in self.critic_name:
                print('{} and its Monte-Carlo Tree Search state-action values'.format(self.critic_name)) 
                def Monte_Carlo():
                    enc_outputs, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs, self.seq_len)
                    _, hiddens, _ = \
                        decode(self.cell, enc_state, self.embedding, \
                               self.decoder_inputs, self.buckets[-1][1], \
                               feed_prev=False)
                    N = 10
                    rewards = []
                    for step in range(self.buckets[-1][1]):
                        each_step_reward = []
                        for _ in range(N):
                            with variable_scope.variable_scope(
                                variable_scope.get_variable_scope(), reuse=True):
                                _, MC_sample, _ = \
                                    decode(self.cell, hiddens[step], self.embedding, \
                                           [self.decoder_inputs[step+1]], self.buckets[-1][1]-step-1, \
                                           feed_prev=True, loop_function=sample_loop_function)
                                all_r, _ = self.critic.discriminator(self.encoder_inputs, self.seq_len, self.decoder_inputs[1:(step+2)]+MC_sample, batch_size)
                            uniW = uniform_weights(self.decoder_inputs[1:(step+2)]+MC_sample)
                            r = get_eos_value(all_r, uniW)
                            each_step_reward.append(tf.reshape(r,[-1]))
                        rewards.append(math_ops.add_n(each_step_reward) / N)
                        #rewards.append(list(each_step_reward))
                    return rewards
                for_D_score = get_eos_value(each_prob, real_uniW)
                for_D_each_prob = Monte_Carlo()
            else:
                for_D_each_prob, for_D_credits, _ = \
                    weighted_rewards(each_prob, self.critic.real_data, real_uniW, 'uniform')
                for_D_score = math_ops.add_n(for_D_each_prob) / (for_D_credits + 1e-12)
                
            if 'seq' in self.critic_name:
                uni_each_value, _, _ = \
                    weighted_rewards(each_value_logit, self.critic.real_data, real_uniW, 'uniform')
            else:
                uni_each_value, _, _ = \
                    weighted_rewards(each_value_prob, self.critic.real_data, real_uniW, 'uniform')

            # FIXME it's usable but a little chaostic
            self.reward = tf.reduce_mean(for_D_score)
            #self.D_probs = [[r[b] for r in for_D_each_prob] for b in range(1)]
            self.D_probs = for_D_each_prob
            self.uniW = uni_each_value#[[w[b] for w in uni_each_value] for b in range(1)]

            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='valuenet')
            params = tf.trainable_variables()
            s2s_params = [ x for x in params if x not in critic_params and x not in value_params ]

            self.pre_saver = tf.train.Saver(var_list=critic_params, sharded=True)
            self.pre_V_saver = tf.train.Saver(var_list=value_params, sharded=True)
            self.pre_s2s_saver = tf.train.Saver(var_list=s2s_params, sharded=True)

        # whole seq2seq saver
        self.saver = tf.train.Saver(max_to_keep=None, sharded=True)

    def train_step(
            self,
            sess,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            bucket_id,
            encoder_lens=None,
            forward=False,
            decoder_outputs=None,#for REINFORCE, MIXER
            rewards=None,#for REINFORCE, MIXER
            GAN_mode=None#for GAN
    ):
    
        #MLE
        if self.critic is None:
            batch_size = encoder_inputs[0].shape[0]
            encoder_size, decoder_size = self.buckets[bucket_id]
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.seq_len] = encoder_lens
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)

            if forward:
                output_feed = [self.losses[bucket_id], self.outputs[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]
            else:
                output_feed = [self.losses[bucket_id], self.op_update[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]

        #SeqGAN, MaliGAN, REGS, ESGAN, EBESGAN
        elif GAN_mode:
            batch_size = encoder_inputs[0].shape[0]
            encoder_size, decoder_size = self.buckets[bucket_id]
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.seq_len] = encoder_lens
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)

            if GAN_mode == 'D':
                for l in range(decoder_size-1):
                    input_feed[self.critic.real_data[l].name] = decoder_inputs[l+1]
                input_feed[self.critic.real_data[decoder_size-1].name] = \
                    np.zeros([batch_size], dtype = np.int32)
                if forward:
                    output_feed = [self.D_losses[bucket_id],
                                   self.losses[bucket_id],
                                   self.D_real[bucket_id],
                                   self.outputs[bucket_id],
                                   self.D_fake[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
                else:
                    output_feed = [self.D_losses[bucket_id],
                                   self.D_solver[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1]

            elif GAN_mode == 'V':
                for l in range(decoder_size-1):
                    input_feed[self.critic.real_data[l].name] = decoder_inputs[l+1]
                input_feed[self.critic.real_data[decoder_size-1].name] = \
                    np.zeros([batch_size], dtype = np.int32)
                output_feed = [self.value_losses[bucket_id],
                               self.v_solver[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]

            else:
                if forward:
                    output_feed = [self.losses[bucket_id],
                                   self.outputs[bucket_id],
                                   self.D_fake[bucket_id],
                                   self.each_probs[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1], outputs[2], outputs[3]
                else:
                    output_feed = [self.losses[bucket_id],
                                   self.perp[bucket_id],
                                   self.D_fake[bucket_id],
                                   self.op_update[bucket_id],
                                   self.outputs[bucket_id],#TODO
                                   self.debug1[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1], outputs[2], outputs[4], outputs[5] #TODO
            
        #REINFORCE
        else:
            batch_size = encoder_inputs[0].shape[0]
            encoder_size, decoder_size = self.buckets[bucket_id]
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.seq_len] = encoder_lens
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)
            if forward:
                output_feed = [self.outputs[bucket_id], self.tmps[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]
            else:
                for l in range(decoder_size-1):
                    input_feed[self.decoder_inputs[l+1].name] = decoder_outputs[l]
                for l in range(decoder_size):
                    input_feed[self.fed_samples[l].name] = decoder_outputs[l]
                input_feed[self.fed_rewards.name] = rewards
                output_feed = [self.losses[bucket_id],
                               self.perp[bucket_id], 
                               self.out_dists[bucket_id],
                               self.op_update[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1], outputs[2]

    def dynamic_decode(self, sess, encoder_inputs, encoder_lens, decoder_inputs, mode='argmax'):
        encoder_size = self.buckets[-1][0]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        input_feed[self.decoder_inputs[0].name] = decoder_inputs[0]
        if mode == 'argmax':
            #output_feed = [self.tmp_outputs[0], self.max_log_prob, self.prob_outputs[0]]
            output_feed = [self.tmp_outputs[0], self.max_log_prob]
        elif mode == 'sample':
            output_feed = [self.samples, self.log_prob]
            #output_feed = [self.samples, self.log_prob, self.reward]
        return sess.run(output_feed, input_feed)

    def test_discriminator(self, sess, encoder_inputs, encoder_lens, decoder_inputs):
        encoder_size, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        input_feed[self.decoder_inputs[0].name] = [data_utils.GO_ID]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l+1].name] = decoder_inputs[l]
            input_feed[self.critic.real_data[l].name] = decoder_inputs[l]
        output_feed = [self.reward, self.D_probs, self.uniW]
        return sess.run(output_feed, input_feed)

    def stepwise_test_beam(self, sess, encoder_inputs, encoder_lens, decoder_inputs):
        encoder_size, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        output_feed = [self.outs]
        return sess.run(output_feed, input_feed)

    def lm_prob(self, sess, decoder_inputs):
        _, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        output_feed = [self.lm_outs]
        return sess.run(output_feed, input_feed)
