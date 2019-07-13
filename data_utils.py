# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import random

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
#_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
#_WORD_SPLIT = re.compile(b"([,!?\":;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend([space_separated_fragment])
        #words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def maybe_split(data_path):
    data_dir, file_name = data_path.rsplit('/',1)
    train_path = os.path.join(data_dir, 'train_' + file_name)
    dev_path = os.path.join(data_dir, 'dev_' + file_name)
    test_path = os.path.join(data_dir, 'test_' + file_name)
    if not (gfile.Exists(train_path)) and ( not gfile.Exists(dev_path) ) \
          and not (gfile.Exists(test_path)):
        if not (gfile.Exists(data_path)):
            raise ValueError("Source file %s not found.", data_path)
        # shuffle data examples
        with gfile.GFile(data_path, mode='r') as f:
            lines = f.readlines()
            ids_list = list(range(0, round(len(lines)/2)))
            random.shuffle(ids_list)
        # parse to train, dev, and test by 8:1:1 portion
        with gfile.GFile(train_path, mode='w') as f:
            for ids in ids_list[:-round(len(ids_list)*0.2)]:
                f.write(lines[2*ids])
                f.write(lines[2*ids+1])
        with gfile.GFile(dev_path, mode='w') as f:
            for ids in ids_list[-round(len(ids_list)*0.2):-round(len(ids_list)*0.1)]:
                f.write(lines[2*ids])
                f.write(lines[2*ids+1])
        with gfile.GFile(test_path, mode='w') as f:
            for ids in ids_list[-round(len(ids_list)*0.1):]:
                f.write(lines[2*ids])
                f.write(lines[2*ids+1])
        #raise ValueError("Train file or development file not found.")
    return (train_path, dev_path)

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().encode('utf-8') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, data_path, vocabulary_size, tokenizer=None):
    # Get data to the specified directory.
    train_path, dev_path = maybe_split(data_path)
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    create_vocabulary(vocab_path, train_path, vocabulary_size, tokenizer, normalize_digits=False)
    # Create token ids for the training data.
    train_ids_path = train_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(train_path, train_ids_path, vocab_path, tokenizer, normalize_digits=False)
    # Create token ids for the development data.
    dev_ids_path = dev_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dev_path, dev_ids_path, vocab_path, tokenizer, normalize_digits=False)
    return (train_ids_path, dev_ids_path, vocab_path)
