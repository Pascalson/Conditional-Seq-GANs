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

def test():
    
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TEST',
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
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=True,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)

        sess.run(tf.variables_initializer(tf.global_variables()))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        print(FLAGS.model_dir)
        print(ckpt)
        model.pre_saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        def beam_search(encoder_inputs, encoder_len):
            beam_size = 10
            beam_len = 1
            beam_encoder_inputs = []
            for l in range(_buckets[-1][0]):
                beam_encoder_inputs.append(
                    np.array([encoder_inputs[l][0] for _ in range(beam_size)], np.int32))
            beam_encoder_len = [encoder_len[0] for _ in range(beam_size)]
            decoder_inputs = [[data_utils.GO_ID]] + [[data_utils.PAD_ID] for _ in range(_buckets[-1][1]-1)]
            outs = model.stepwise_test_beam(sess, encoder_inputs, encoder_len, decoder_inputs)
            outs = outs[0][0]
            top_k_idx = np.argsort(outs[0])[-beam_size:]
            path = [[data_utils.GO_ID for _ in range(beam_size)], top_k_idx] + \
                   [[data_utils.PAD_ID for _ in range(beam_size)] for _ in range(_buckets[-1][1]-beam_len-1)]
            end_path = []
            probs = np.zeros(beam_size)
            for k in range(beam_size):
                probs[k] += np.log(outs[0][top_k_idx[k]])
                end_path.append(data_utils.EOS_ID == path[-1][k])

            for _ in range(_buckets[-1][1]-1):
                beam_len += 1
                outs = model.stepwise_test_beam(sess, beam_encoder_inputs, beam_encoder_len, path)
                outs = outs[0][beam_len-1]
                top_k_idxes = [np.argsort(outs[p])[-beam_size:] for p in range(beam_size)]
                edges = []
                tmp_probs = []
                for p in range(beam_size):
                    if end_path[p]:
                        tmp_probs.extend([probs[p]])
                        edges.append(1)
                    else:
                        tmp_probs.extend([probs[p]+np.log(outs[p][k]) for k in top_k_idxes[p]])
                        edges.append(beam_size)
                top_k_path_idx = np.argsort(tmp_probs)[-beam_size:]
                edges_scale = [sum(edges[:i+1]) for i in range(beam_size)]
                tmp_path = []
                for l in range(beam_len):
                    step = []
                    for k in top_k_path_idx:
                        path_id = min([i for i in range(beam_size) if edges_scale[i] > k])
                        step.append(path[l][path_id])
                    tmp_path.append(step)
                path = tmp_path
                step = []
                expand_edges_scale = [0] + edges_scale
                for k in top_k_path_idx:
                    path_id = min([i for i in range(beam_size) if edges_scale[i] > k])
                    step_id = k - expand_edges_scale[path_id]
                    step.append(top_k_idxes[path_id][step_id])
                path.append(step)
                for i, k in enumerate(top_k_path_idx):
                    probs[i] = tmp_probs[k]
                    end_path[i] = False
                    for l in range(beam_len+1):
                        if path[l][i] == data_utils.EOS_ID:
                            end_path[i] = True
                path += [[data_utils.PAD_ID for _ in range(beam_size)] for _ in range(_buckets[-1][1]-beam_len-1)]
            return path, probs

        def MMI(decoder_inputs):
            lm_probs = model.lm_prob(sess, decoder_inputs)
            lm_probs = lm_probs[0]
            lm_prob = []
            lens = []
            for p in range(len(decoder_inputs[0])):
                tmp_prob = 0.0
                for l in range(len(decoder_inputs)-1):
                    if l < 3:
                        tmp_prob += np.log(lm_probs[l][p][decoder_inputs[l+1][p]])
                    if decoder_inputs[l+1][p] == data_utils.EOS_ID:
                        lens.append(l)
                        break
                lm_prob.append(tmp_prob)
                if len(lens) < len(lm_prob):
                    lens.append(len(decoder_inputs)-1)
            return lm_prob, lens


        if FLAGS.test_type == 'accuracy':
            grammar_critic = load_critic(FLAGS.test_critic)
            argmax_correct, argmax_count = 0, 0
            sample_correct, sample_count = 0, 0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            with open(test_data,'r') as f, open('accuracy_hist_samples.txt','w') as fout:
                all_f = f.readlines()#for data only involves inputs
                
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)

                batch_encoder_inputs = []
                batch_encoder_lens = []
                batch_sentences = []
                batch_flag = 0
                for _ in range(int(len(tmp_encoder_inputs) / FLAGS.batch_size)):
                    encoder_inputs = []
                    for idx in range(_buckets[-1][0]):
                        encoder_inputs.append(
                            np.array([tmp_encoder_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(FLAGS.batch_size)],
                                     dtype = np.int32))
                    batch_encoder_inputs.append(encoder_inputs)
                    batch_encoder_lens.append(tmp_encoder_lens[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_sentences.append(all_f[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_flag += FLAGS.batch_size
                decoder_inputs = [[data_utils.GO_ID for _ in range(FLAGS.batch_size)]]
                print('batch number:{}'.format(len(batch_encoder_inputs)))

                for batch_idx, (encoder_inputs, encoder_lens) in \
                        enumerate(zip(batch_encoder_inputs, batch_encoder_lens)):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, \
                                                      encoder_lens, \
                                                      decoder_inputs)
                    for idx, sentence in enumerate(all_f[batch_idx*FLAGS.batch_size:(batch_idx+1)*FLAGS.batch_size]):
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        if grammar_critic.check_ans(sentence.split(), ans):
                            argmax_correct += 1
                        argmax_count += 1


                hist_samples = {}
                coverage = 0.0
                for _ in range(100):
                    for batch_idx, (encoder_inputs, encoder_lens) in \
                            enumerate(zip(batch_encoder_inputs, batch_encoder_lens)):
                        samples, log_prob = model.dynamic_decode(sess, encoder_inputs, \
                                                                 encoder_lens, \
                                                                 decoder_inputs, mode='sample')
                        for idx, sentence in enumerate(batch_sentences[batch_idx]):
                            sample = [sample_ids[idx] for sample_ids in samples]
                            if data_utils.EOS_ID in sample:
                                sample = sample[:sample.index(data_utils.EOS_ID)]
                            ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in sample])

                            if grammar_critic.check_ans(sentence.split(), ans):
                                sample_correct += 1
                                if sentence.strip() not in hist_samples:
                                    hist_samples[sentence.strip()] = []
                                if (ans,'T') not in hist_samples[sentence.strip()]:
                                    hist_samples[sentence.strip()].append((ans,'T'))
                                    coverage += 1.0 / grammar_critic.possible_ans_num(sentence.split())
                            sample_count += 1


                print('argmax accuracy: {}'.format(float(argmax_correct) / argmax_count))
                print('sample accuracy: {}'.format(float(sample_correct) / sample_count))
                print('sample coverage: {}'.format(coverage / len(tmp_encoder_inputs)))
                print('number of inputs (regardless to batch):{}'.format(len(tmp_encoder_inputs)))
                for key, value in hist_samples.items():
                    fout.write("{}:{}\n".format(key, value))

        else:
            sys.stdout.write('> ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                if sentence.strip() == 'exit()':
                    break
                # step
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                encoder_lens = [len(token_ids)]
                token_ids = list(reversed(token_ids)) + encoder_pad
                encoder_inputs = []
                for idx in token_ids:
                    encoder_inputs.append([idx])
                decoder_inputs = [[data_utils.GO_ID]]
                
                if FLAGS.test_type == 'realtime_argmax':#TODO: check WHO don't have argmax?
                    outs, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs)
                    #outs = [int(np.argmax(logit, axis=1)) for logit in outs]
                    
                    outputs = [output_ids[0] for output_ids in outs]
                    #prob_outputs = [prob[0] for prob in probs]
                    if data_utils.EOS_ID in outputs:
                        idx = outputs.index(data_utils.EOS_ID)
                        #outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        outputs = outputs[:idx]
                        #prob_outputs = prob_outputs[:idx]
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                    #print(prob_outputs)
                    print("log prob %.6f" % log_prob[0])

                elif FLAGS.test_type == 'realtime_beam_search':
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    for b in range(len(outs[0])):
                        outputs = [output_ids[b] for output_ids in outs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                        print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                        print("log prob %.6f" % log_prob[b])

                elif FLAGS.test_type == 'realtime_MMI':
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    lm_log_prob, lens = MMI(outs)
                    MMI_scores = np.array(log_prob) - 0.5 * np.array(lm_log_prob) + 0.5 * np.array(lens)
                    MMI_idx = np.argsort(MMI_scores)
                    for b in [MMI_idx[-1]]:#range(len(outs[0])):
                        outputs = [output_ids[b] for output_ids in outs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                        print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                        print("log prob %.6f" % log_prob[b])
                        print("general log prob(w/o condition) %.6f" % lm_log_prob[b])
                        print("sentence lenth %d" % lens[b])

                elif FLAGS.test_type == 'realtime_sample':
                    samples, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs, mode='sample')
                    sample = [sample_ids[0] for sample_ids in samples]
                    if data_utils.EOS_ID in sample:
                        sample = sample[:sample.index(data_utils.EOS_ID)]
                    print(sample)
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in sample]))
                    print("log prob %.6f" % log_prob)
                
                sys.stdout.write('> ')
                sys.stdout.flush()
                sentence = sys.stdin.readline()
