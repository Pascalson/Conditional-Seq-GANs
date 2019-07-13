import os
from random import shuffle
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from scipy.special import comb
import numpy as np
import random
import math

import data_utils

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class ValueNet:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'ValueNet'
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])
        self.W = tf.Variable(xavier_init([size*num_layers, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1]))
        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, batch_size, dtype=tf.float32):
        # notice reversed parts
        with variable_scope.variable_scope('valuenet') as scope:
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp, sequence_length=inp_lens, dtype=dtype)
            prob, logit = self.decode(inp_state, ans)
            return prob, logit

    def decode(self, init_state, decoder_inputs):
        logits = []
        probs = []
        state = init_state
        emb_inputs = (embedding_ops.embedding_lookup(self.embedding, i)
                      for i in decoder_inputs)
        for i, emb_inp in enumerate(emb_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            state_vec = tf.concat(state, 1)
            logits.append(tf.matmul(state_vec, self.W)+self.b)
            probs.append(tf.nn.sigmoid(logits[-1]))
            # notice : the order is different from GAN
            output, state = self.cell(emb_inp, state)
        return probs, logits

class StepGAN:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'StepGAN'
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])
        self.D_W = tf.Variable(xavier_init([size*num_layers, 1]))
        self.D_b = tf.Variable(tf.zeros(shape=[1]))
        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, batch_size, dtype=tf.float32):
        # notice reversed parts
        with variable_scope.variable_scope('critic') as scope:
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp, sequence_length=inp_lens, dtype=dtype)
            D_prob, D_logit = self.decode(inp_state, ans)
            return D_prob, D_logit

    def decode(self, init_state, decoder_inputs):
        logits = []
        probs = []
        state = init_state
        emb_inputs = (embedding_ops.embedding_lookup(self.embedding, i)
                      for i in decoder_inputs)
        for i, emb_inp in enumerate(emb_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = self.cell(emb_inp, state)
            state_vec = tf.concat(state, 1)
            logits.append(tf.matmul(state_vec, self.D_W)+self.D_b)
            probs.append(tf.nn.sigmoid(logits[-1]))
        return probs, logits


################################
# Generate Scores for REINFORCE
# Tasks: Counting
################################

class Counting_Task:
    def __init__(self):
        self.UPBOUND = 9
        self.SEQ_LEN = 10
        vocab_path = 'data/counting/vocab{}'.format(self.UPBOUND + 1 + 4)
        vocab, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        self.number_rev_vocab = tf.string_to_number(tf.constant(self.rev_vocab[4:]), tf.int32)

        def gen_data(f, num_data, test=False):
            for _ in range(num_data):
                inp_len = np.random.randint(1, high=self.SEQ_LEN)
                inp = np.random.randint(self.UPBOUND+1, size=inp_len)           
                buf = ' '.join(str(i) for i in inp)
                buf += '\n'
                f.write(buf)
                if not test:
                    #out_flags_num = np.random.randint(inp_len + 1)
                    out_flag = np.random.randint(inp_len)
                    out = [out_flag, inp[out_flag], len(inp) - out_flag - 1]
                    buf = ' '.join(str(i) for i in out)
                    buf += '\n'
                    f.write(buf)

        if not os.path.exists('data/counting/train_counting.txt'):
            with open('data/counting/train_counting.txt', 'w') as f:
                gen_data(f, 100000)
            with open('data/counting/dev_counting.txt', 'w') as f:
                gen_data(f, 10000)
            with open('data/counting/test_counting.txt', 'w') as f:
                gen_data(f, 10000, test=True)

    def possible_ans_num(self, inp):
        return len(inp)

    def get_ans_space(self):#all possible action space
        ans_space = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    ans = str(i)+' '+str(j)+' '+str(k)
                    ans_space.append(ans)
        return ans_space

    def gen_ans(self, inp):
        inp = inp.split()
        ans_set = []
        for l in range(len(inp)):
            ans = str(l)+' '+str(inp[l])+' '+str(len(inp)-l-1)
            ans_set.append(ans)
        return ans_set

    def check_ans(self, inp, ans):
        ans = ans.split()
        try:
            if len(ans) == 3:
                if ans[0] == "_UNK" and ans[2] != "_UNK":
                    if int(ans[2]) < len(inp) and len(inp) - int(ans[2]) > 9:
                        if inp[-int(ans[2])-1] == ans[1]:
                            return True
                elif ans[0] != "_UNK" and ans[2] == "_UNK":
                    if int(ans[0]) < len(inp) and len(inp) - int(ans[0]) > 9:
                        if inp[int(ans[0])] == ans[1]:
                            return True
                elif int(ans[0]) + int(ans[2]) + 1 == len(inp) and int(ans[0]) >= 0:
                    if ans[1] == inp[int(ans[0])]:
                        return True
        except ValueError:
            return False
        return False
