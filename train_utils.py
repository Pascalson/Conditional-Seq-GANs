import tensorflow as tf
import numpy as np
import pickle
import random
import re
import os
import sys

import data_utils

import args
FLAGS = args.FLAGS
_buckets = args._buckets

def read_data(data_path, maxlen, max_size=None):
    dataset = []
    with tf.gfile.GFile(data_path, mode='r') as data_file:
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and \
                len(source.split()) < maxlen and len(target.split())+1 < maxlen and \
                (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(EOS_ID)
            dataset.append([source_ids, target_ids])
            source = data_file.readline()
            target = data_file.readline()
    return dataset

def read_data_with_buckets(data_path, max_size=None):
    buckets = _buckets
    dataset = [[] for _ in buckets]
    with tf.gfile.GFile(data_path, mode='r') as data_file:
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and \
                (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            # form dataset
            stored = 0
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    dataset[bucket_id].append([source_ids, target_ids])
                    stored = 1
                    break
            if stored == 0:#truncate the length
                dataset[-1].append([ source_ids[:buckets[-1][0]], target_ids[:buckets[-1][1]] ])
            # next loop
            source = data_file.readline()
            target = data_file.readline()
    return dataset

def get_batch_with_buckets(data, batch_size, bucket_id, size=None):
    # data should be [whole_data_length x (source, target)] 
    # decoder_input should contain "GO" symbol and target should contain "EOS" symbol
    encoder_size, decoder_size = _buckets[bucket_id]
    encoder_inputs, decoder_inputs, seq_len = [], [], []

    for i in range(batch_size):
        encoder_input, decoder_input = random.choice(data[bucket_id])
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)
        seq_len.append(len(encoder_input))
        decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input))
        decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    # make batch for encoder inputs
    for length_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
    # make batch for decoder inputs
    for length_idx in range(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
        batch_weight = np.ones(batch_size, dtype = np.float32)
        for batch_idx in range(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, seq_len, encoder_inputs

def check_batch_ans(mycritic, inp, inp_lens, ans):
    inp = [[inp[t][i] for t in reversed(range(inp_lens[i]))] for i in range(len(inp[0]))]
    ans = [[ans[t][i] for t in range(len(ans))] for i in range(len(ans[0]))]
    rewards = []
    for per_inp, per_ans in zip(inp, ans):
        per_inp = [ tf.compat.as_str(mycritic.rev_vocab[out]) for out in per_inp ]
        if data_utils.EOS_ID in per_ans:
            per_ans = per_ans[:per_ans.index(data_utils.EOS_ID)]
        per_ans = ' '.join(tf.compat.as_str(mycritic.rev_vocab[out]) for out in per_ans)
        if mycritic.check_ans(per_inp, per_ans):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
