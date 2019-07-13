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

def train_mle():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    def build_summaries():
        train_loss = tf.Variable(0.)
        tf.summary.scalar("train_loss", train_loss)
        eval_losses = []
        for ids, _ in enumerate(_buckets):
            eval_losses.append(tf.Variable(0.))
            tf.summary.scalar("eval_loss_{}".format(ids), eval_losses[ids])
        summary_vars = [train_loss] + eval_losses
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    # parse data and build vocab if there do not exist one.
    train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)
        # build summary (log)
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(FLAGS.model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
        if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
            model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)
    
        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        dev_set = read_data_with_buckets(dev)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # main process
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # get batch from a random selected bucket
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            # each training step
            start_time = time.time()
            step_loss, _ = model.train_step(sess, encoder_inputs, \
                                            decoder_inputs, weights, \
                                            bucket_id, seq_lens)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            # log, save and eval
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print ("global step %d; learning rate %.4f; step-time %.2f; perplexity "
                       "%.2f; loss %.2f"
                       % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.op_lr_decay)
                previous_losses.append(loss)
                # eval
                eval_losses = []
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        eval_losses.append(0.)
                        continue
                    encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                        get_batch_with_buckets(dev_set, FLAGS.batch_size, bucket_id)
                    eval_loss, outputs = model.train_step(sess, encoder_inputs, \
                                                    decoder_inputs, weights, \
                                                    bucket_id, seq_lens, forward=True)
                    """
                    # for seeing the current outputs
                    outputs = [output_ids[0] for output_ids in outputs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    print(outputs)
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                    """
                    eval_losses.append(eval_loss)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.2f; loss %.2f" % (bucket_id, eval_ppx, eval_loss))
                # write summary
                feed_dict = {}
                for ids, key in enumerate(summary_vars[1:]):
                    feed_dict[key] = eval_losses[ids]
                feed_dict[summary_vars[0]] = loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_step.eval())
                writer.flush()
                # Save checkpoint and zero timer and loss.
                ckpt_path = os.path.join(FLAGS.model_dir, "ckpt")
                model.saver.save(sess, ckpt_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()
