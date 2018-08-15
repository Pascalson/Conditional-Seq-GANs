import tensorflow as tf
import numpy as np
import pickle
import random
import re
import os
import sys
import time
import math

from seq2seq_model_comp import *
import data_utils

from train_utils import *
import args
FLAGS = args.FLAGS
_buckets = args._buckets

def train_gan():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    def build_summaries():
        loss = tf.Variable(0.)
        tf.summary.scalar("loss", loss)
        perp = tf.Variable(0.)
        tf.summary.scalar("perp", perp)
        reward = tf.Variable(0.)
        tf.summary.scalar("reward", reward)
        if 'GAN' in FLAGS.gan_type or FLAGS.gan_type == 'REGS':
            D_loss = tf.Variable(0.)
            tf.summary.scalar("D_loss", D_loss)
            summary_vars = [loss, perp, reward, D_loss]
        else:
            summary_vars = [loss, perp, reward]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    # parse data and build vocab if there do not exist one.
    train, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    reward_out_path = os.path.join(FLAGS.model_dir, 'rewards_trajectory.txt')
    
    with tf.Session() as sess, open(reward_out_path,'w') as f_reward_out:
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            other_option=FLAGS.option,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            D_lr=FLAGS.D_lr,
            D_lr_decay_factor=FLAGS.D_lr_decay_factor,
            v_lr=FLAGS.v_lr,
            v_lr_decay_factor=FLAGS.v_lr_decay_factor,
            dtype=tf.float32)
        # build summary and initialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        value_path = os.path.join(FLAGS.pre_D_model_dir, '..', 'value')
        log_dir = os.path.join(FLAGS.model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        elif os.path.exists(FLAGS.pre_model_dir):
            pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
            if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
                print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
                model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)
            else:
                print ('no previous model, create a new one')
            if os.path.exists(FLAGS.pre_D_model_dir):
                pre_D_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
                if pre_D_ckpt and tf.train.checkpoint_exists(pre_D_ckpt.model_checkpoint_path):
                    print ('read in model from {}'.format(pre_D_ckpt.model_checkpoint_path))
                    model.pre_D_saver.restore(sess, pre_D_ckpt.model_checkpoint_path)
                else:
                    print ('no previous critic, create a new one')
            if os.path.exists(value_path):
                pre_V_ckpt = tf.train.get_checkpoint_state(value_path)
                if pre_V_ckpt and tf.train.checkpoint_exists(pre_V_ckpt.model_checkpoint_path):
                    print ('read in model from {}'.format(pre_V_ckpt.model_checkpoint_path))
                    model.pre_value_saver.restore(sess, pre_V_ckpt.model_checkpoint_path)
                else:
                    print ('no previous critic, create a new one')

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        if FLAGS.option == 'MIXER':# in MIXER, using the longest bucket
            train_buckets_sizes = [len(train_set[-1])]
        else:# in REINFORCE, SeqGAN, using buckets, (or can set a longest bucket)
            train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        D_step = FLAGS.D_step
        G_step = FLAGS.G_step
        # for debug
        avg_debug1 = 0.0
        # main process
        step_time, loss, reward = 0.0, 0.0, 0.0
        perp, D_loss = 0.0, 0.0
        bucket_times = [0 for _ in range(len(_buckets))]
        #value_loss = 0.0
        s = _buckets[-1][1]
        current_step = 0
        previous_rewards = []
        np.random.seed(1234)
        while True:
            if FLAGS.option == 'MIXER':
                bucket_id = -1
            else:
                # get batch from a random selected bucket
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            # each training step
            start_time = time.time()
            if hasattr(model.critic, 'discriminator'):
                for _ in range(D_step):# D steps
                    step_D_loss, _ = model.train_step(sess, encoder_inputs, \
                                                      decoder_inputs, weights, \
                                                      bucket_id, seq_lens, GAN_mode='D')
                    random_number_01 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(train_buckets_scale))
                                     if train_buckets_scale[i] > random_number_01])
                    encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                        get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)
                    D_loss += step_D_loss / FLAGS.steps_per_checkpoint / D_step
                # value net, only 1 iteration
                _, _ = model.train_step(sess, encoder_inputs, \
                                        decoder_inputs, weights, \
                                        bucket_id, seq_lens, GAN_mode='V')
                for _ in range(G_step):# G steps
                    step_loss, step_perp, step_reward, G_outputs, debug1 = \
                        model.train_step(sess, encoder_inputs, \
                                         decoder_inputs, weights, \
                                         bucket_id, seq_lens, \
                                         GAN_mode='G')
                    if 'StepGAN' in FLAGS.gan_type or FLAGS.gan_type == 'REGS' or FLAGS.gan_type == 'MaskGAN':
                        step_reward = np.concatenate((step_reward, \
                                        np.zeros((FLAGS.batch_size, _buckets[-1][1] - _buckets[bucket_id][1]))),
                                        axis=1)
                    bucket_times[bucket_id] += 1
                    # each time get another batch
                    random_number_01 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(train_buckets_scale))
                                     if train_buckets_scale[i] > random_number_01])
                    encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                        get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)
                    #TODO notice the relationship between steps_per_checkpoint and G steps
                    reward += np.sum(step_reward, axis=0) / FLAGS.batch_size / G_step
                    avg_debug1 += debug1 / FLAGS.batch_size / G_step

            #TODO: use py_func, REINFORCE
            else:
                step_samples, _ = model.train_step(sess, encoder_inputs, \
                                                   decoder_inputs, weights, \
                                                   bucket_id, seq_lens, forward=True)
                step_reward = check_batch_ans(model.critic, encoder_inputs, seq_lens, step_samples)
                step_loss, step_perp, _ = \
                    model.train_step(sess, encoder_inputs, \
                                     decoder_inputs, weights, \
                                     bucket_id, seq_lens, \
                                     decoder_outputs=step_samples, \
                                     rewards=step_reward)
                reward += sum(step_reward) / FLAGS.batch_size / FLAGS.steps_per_checkpoint
                
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += sum(sum(step_loss)) / FLAGS.batch_size / FLAGS.steps_per_checkpoint
            perp += step_perp / FLAGS.steps_per_checkpoint
            # log, save and eval
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                #print(value)
                print ("global step %d; learning rate %.8f; D lr %.8f; step-time %.2f;"
                       % (model.global_step.eval(),
                          model.learning_rate.eval(),
                          model.D_lr.eval(),
                          step_time))
                #print(avg_debug1)
                print("perp %.4f" % perp)
                #print("loss %.8f" % loss)
                print(loss)
                if hasattr(model.critic, 'discriminator'):
                    print("D-loss %.4f" % D_loss)
                    #print("value-loss %.4f" % value_loss)
                if 'StepGAN' in FLAGS.gan_type or FLAGS.gan_type == 'REGS' or FLAGS.gan_type == 'MaskGAN':
                    len_times = []
                    for k, bucket in enumerate(_buckets):
                        len_times += [sum(bucket_times[k:])+1e-12] * (bucket[1]-len(len_times))
                    reward = np.true_divide(reward, len_times)
                print("reward(D_fake_value) {}".format(reward))
                f_reward_out.write("{}\n".format(reward))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if 'EB' in FLAGS.gan_type:
                    if len(previous_rewards) > 2 and np.sum(reward) > max(previous_rewards[-3:]):
                        sess.run(model.op_lr_decay)
                    elif len(previous_rewards) > 2 and np.sum(reward) < min(previous_rewards[-3:]):
                        sess.run(model.op_D_lr_decay)
                else:
                    if len(previous_rewards) > 2 and np.sum(reward) < min(previous_rewards[-3:]):
                        sess.run(model.op_lr_decay)
                    elif len(previous_rewards) > 2 and np.sum(reward) > max(previous_rewards[-3:]):
                        sess.run(model.op_D_lr_decay)
                previous_rewards.append(np.sum(reward))
                # write summary
                feed_dict = {}
                feed_dict[summary_vars[0]] = loss
                feed_dict[summary_vars[1]] = perp
                feed_dict[summary_vars[2]] = np.sum(reward) / _buckets[-1][1]
                if hasattr(model.critic, 'discriminator'):
                    feed_dict[summary_vars[3]] = D_loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_step.eval())
                writer.flush()
                # Save checkpoint and zero timer and loss.
                ckpt_path = os.path.join(FLAGS.model_dir, "ckpt")
                model.saver.save(sess, ckpt_path, global_step=model.global_step)
                if FLAGS.fix_steps > 0:
                    if model.global_step.eval() >= FLAGS.fix_steps:
                        return
                step_time, loss, reward = 0.0, 0.0, 0.0
                perp, D_loss = 0.0, 0.0
                bucket_times = [0 for _ in range(len(_buckets))]

                sys.stdout.flush()
