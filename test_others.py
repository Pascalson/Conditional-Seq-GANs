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


def test_D():
    with tf.Session() as sess:
        model = Seq2Seq(
            'D_TEST',
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

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        D_ckpt = ckpt#tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
        model.pre_saver.restore(sess, D_ckpt.model_checkpoint_path)
        print ('read in discriminator model from {}'.format(ckpt.model_checkpoint_path))
        #value_path = os.path.join(FLAGS.pre_D_model_dir, '..', 'value')
        V_ckpt = ckpt#tf.train.get_checkpoint_state(value_path)
        model.pre_V_saver.restore(sess, V_ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(V_ckpt.model_checkpoint_path))
        #s2s_path = os.path.join(FLAGS.pre_model_dir, '../MLE')
        s2s_ckpt = ckpt#tf.train.get_checkpoint_state(s2s_path)
        model.pre_s2s_saver.restore(sess, s2s_ckpt.model_checkpoint_path)
        print ('read in generator model from {}'.format(s2s_ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        if FLAGS.test_type == 'batch_test_D':
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join('test_D_' + file_name)
            out_lists = os.path.join(FLAGS.gan_type + 'test_D_' + '.out')
            all_total_score = []
            all_each_scores = []
            all_uniW = []
            with open(test_data,'r') as f:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_Q_inputs = []
                tmp_Q_lens = []
                for _, sentence in enumerate(all_f):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    Q_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_Q_lens.append(len(token_ids))
                    tmp_Q_inputs.append(list(reversed(token_ids)) + Q_pad)
                tmp_A_inputs = []
                tmp_A_lens = []
                for _, sentence in enumerate(all_ref):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    A_pad = [data_utils.PAD_ID] * (_buckets[-1][1] - len(token_ids) - 1)
                    tmp_A_lens.append(len(token_ids))
                    tmp_A_inputs.append(list(token_ids) + [data_utils.EOS_ID] + A_pad)

                batch_Q_inputs = []
                batch_Q_lens = []
                batch_A_inputs = []
                batch_flag = 0
                bs = 1
                for _ in range(int(len(tmp_Q_inputs) / bs)):
                    # Q batch
                    Q_inputs = []
                    for idx in range(_buckets[-1][0]):
                        Q_inputs.append(
                            np.array([tmp_Q_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(bs)],
                                     dtype = np.int32))
                    batch_Q_inputs.append(Q_inputs)
                    batch_Q_lens.append([tmp_Q_lens[batch_flag]])
                    # A batch
                    A_inputs = []
                    for idx in range(_buckets[-1][1]):
                        A_inputs.append(
                            np.array([tmp_A_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(bs)],
                                     dtype = np.int32))
                    batch_A_inputs.append(A_inputs)
                    # batch flag
                    batch_flag += bs

                for Q_inputs, Q_lens, A_inputs in zip(batch_Q_inputs, batch_Q_lens, batch_A_inputs):
                    print(Q_inputs)
                    print(Q_lens)
                    print(A_inputs)
                    total_score, each_scores, uniW = model.test_discriminator(sess, Q_inputs, Q_lens, A_inputs)
                    for idx in range(len(Q_inputs[0])):
                        all_total_score.append(total_score)
                        all_each_scores.append([score[idx] for score in each_scores])
                        all_uniW.append([value[idx] for value in uniW])
                pickle.dump([all_total_score, all_each_scores, all_uniW], open(out_lists,'wb'))

        else:
            sys.stdout.write('> ')
            sys.stdout.flush()
            Q = sys.stdin.readline()
            sys.stdout.write('>> ')
            sys.stdout.flush()
            A = sys.stdin.readline()
            while Q and A:
                if Q.strip() == 'exit()' or A.strip() == 'exit()':
                    break
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(Q), vocab, normalize_digits=False)
                Q_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                Q_lens = [len(token_ids)]
                token_ids = list(reversed(token_ids)) + Q_pad
                Q_inputs = []
                for idx in token_ids:
                    Q_inputs.append([idx])

                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(A), vocab, normalize_digits=False)
                A_pad = [data_utils.PAD_ID] * (_buckets[-1][1] - len(token_ids)-1)
                token_ids = list(token_ids) + [data_utils.EOS_ID] + A_pad
                A_inputs = []
                for idx in token_ids:
                    A_inputs.append([idx])

                whole_score, each_score, uniW = model.test_discriminator(sess, Q_inputs, Q_lens, A_inputs)
                print(whole_score)
                print(each_score)
                print(uniW)

                sys.stdout.write('> ')
                sys.stdout.flush()
                Q = sys.stdin.readline()
                sys.stdout.write('>> ')
                sys.stdout.flush()
                A = sys.stdin.readline()
