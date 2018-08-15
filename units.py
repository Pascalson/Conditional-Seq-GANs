import tensorflow as tf
import data_utils
import collections

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

def encode(cell, encoder_inputs, seq_len=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
        scope.set_dtype(dtype)

        return tf.nn.static_rnn(
            cell,
            encoder_inputs,
            sequence_length=seq_len,
            dtype=dtype)

def decode(cell, init_state, embedding, decoder_inputs, maxlen,
           feed_prev=False, loop_function=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_decoder") as scope:
        outputs = []
        hiddens = []
        state = init_state

        if not feed_prev:
            emb_inputs = (embedding_ops.embedding_lookup(embedding, i)
                          for i in decoder_inputs)
            for i, emb_inp in enumerate(emb_inputs):
                if i >= maxlen:
                    break
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                output, state = cell(emb_inp, state)
                outputs.append(output) 
                hiddens.append(state)
            return outputs, hiddens, state

        else:
            samples = []
            i = 0
            emb_inp = embedding_ops.embedding_lookup(embedding, decoder_inputs[0])
            prev = None
            tmp = None
            while(True):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                output, state = cell(emb_inp, state)
                outputs.append(output)
                hiddens.append(state)
                prev = output
                with tf.variable_scope('loop', reuse=True):
                    if prev is not None:
                        tmp = loop_function(prev)
                if tmp is not None:
                    if isinstance(tmp, list):
                        emb_inp, prev_symbol = tmp
                        samples.append(prev_symbol)
                    else:
                        emb_inp = tmp
                i += 1
                if i >= maxlen:
                    break
            return outputs, samples, hiddens
