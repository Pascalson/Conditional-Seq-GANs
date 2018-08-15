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

from train_utils import * # read data
import args
#import main
#from main import * # global variable
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

            #for _ in range(2-1):
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

        # main testing process.
        if FLAGS.test_type == 'BLEU':
            count = 0
            bleu_1, bleu_2, bleu_3, bleu_4 = 0.0, 0.0, 0.0, 0.0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            #test_fout = os.path.join(FLAGS.model_dir, 'test_argmax_outs.txt')
            #with open(test_data,'r') as f, open(test_fout,'w') as fout:
            with open(test_data,'r') as f:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f[:1000]):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
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

                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    for idx, sentence in enumerate(all_f[batch_idx*FLAGS.batch_size:(batch_idx+1)*FLAGS.batch_size]):
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        #fout.write(sentence)
                        #fout.write(ans)
                        #fout.write('\n')
                        #bleu_2 += sentence_bleu(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), weights=[0.5,0.5,0,0])
                        #bleu_3 += sentence_bleu(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), weights=[0.33,0.33,0.33,0])
                        #bleu_4 += sentence_bleu(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), weights=[0.25,0.25,0.25,0.25])
                        tmp_bleu_1 = modified_precision(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), 1)
                        tmp_bleu_2 = modified_precision(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), 2)
                        #print(tmp_bleu_1)
                        #print(tmp_bleu_2)
                        bleu_1 += tmp_bleu_1#*len(ans.split())
                        bleu_2 += tmp_bleu_2#*len(ans.split())
                        #count += len(ans.split())
                #if bleu_2/count == 0:
                #    BLEU_2 = np.exp(0.5*np.log(bleu_1/count).5*np.log(bleu_2/count))
                #else:
                #    BLEU_2 = np.exp(0.5*np.log(bleu_1/count)+0.5*np.log(bleu_2/count))
                #print('BLEU-2:{}'.format(BLEU_2))
                print('UNI:{}'.format(bleu_1))
                print('BI:{}'.format(bleu_2))

        elif FLAGS.test_type == 'per_print':
            beam_bleu_1, mmi_bleu_1 = 0.0, 0.0
            beam_bleu_2, mmi_bleu_2 = 0.0, 0.0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            #test_data = os.path.join(data_dir, 'test_' + file_name)
            test_data = os.path.join('other_test_' + file_name)
            bs_output_path = os.path.join(FLAGS.model_dir, 'other_test_per_BS.txt')
            mmi_output_path = os.path.join(FLAGS.model_dir, 'other_test_per_MMI.txt')
            with open(test_data,'r') as f, open(bs_output_path,'w') as bs_fout, open(mmi_output_path,'w') as mmi_fout:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                for idx, sentence in enumerate(all_f):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    encoder_lens = [len(token_ids)]
                    token_ids = list(reversed(token_ids)) + encoder_pad
                    encoder_inputs = []
                    for idx in token_ids:
                        encoder_inputs.append([idx])
                    decoder_inputs = [[data_utils.GO_ID]]
                    bs_fout.write(sentence)
                    mmi_fout.write(sentence)
                    """
                    # greedy
                    outs, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs)
                    outputs = [output_ids[0] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    greedy = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
                    greedy_bleu_2 += \
                        sentence_bleu(all_ref[idx].split(), \
                                      greedy.split(), weights=[0.5,0.5,0,0])
                    fout.write(greedy+",")
                    """
                    # beam_search
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    outputs = [output_ids[-1] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                    beam = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
                    bs_fout.write(beam+"\n")
                    #beam_bleu_1 += \
                    #        modified_precision(all_ref[idx].split(), beam.split(), 1)
                    #beam_bleu_2 += \
                    #        modified_precision(all_ref[idx].split(), beam.split(), 2)
                    # MMI
                    lm_log_prob, lens = MMI(outs)
                    MMI_scores = np.array(log_prob) - 0.5 * np.array(lm_log_prob) + 0.5 * np.array(lens)
                    MMI_idx = np.argsort(MMI_scores)
                    outputs = [output_ids[MMI_idx[-1]] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                    mmi = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
                    mmi_fout.write(mmi+"\n")
                    #mmi_bleu_1 += \
                    #        modified_precision(all_ref[idx].split(), mmi.split(), 1)
                    #mmi_bleu_2 += \
                    #        modified_precision(all_ref[idx].split(), mmi.split(), 2)
                #print('beam search, BLEU-1:{}\n'.format(beam_bleu_1))
                #print('beam search, BLEU-2:{}\n'.format(beam_bleu_2))
                #print('MMI, BLEU-1:{}'.format(mmi_bleu_1))
                #print('MMI, BLEU-2:{}'.format(mmi_bleu_2))

        elif FLAGS.test_type == 'accuracy':
            grammar_critic = load_critic(FLAGS.test_critic)
            argmax_correct, argmax_count = 0, 0
            sample_correct, sample_count = 0, 0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            with open(test_data,'r') as f, open('accuracy_hist_samples.txt','w') as fout:
                all_f = f.readlines()
                #all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                #all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
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
                print(len(batch_encoder_inputs))

                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    for idx, sentence in enumerate(all_f[batch_idx*FLAGS.batch_size:(batch_idx+1)*FLAGS.batch_size]):
                        #outputs = [int(np.argmax(logit, axis=1)) for logit in outputs]
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        #if ans in mydata.seq_abbrev(sentence):
                        if grammar_critic.check_ans(sentence.split(), ans):
                            argmax_correct += 1
                        argmax_count += 1

                hist_samples = {}
                coverage = 0.0
                for _ in range(100):
                    for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                        samples, log_prob = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                                 decoder_inputs, mode='sample')
                        for idx, sentence in enumerate(batch_sentences[batch_idx]):
                            sample = [sample_ids[idx] for sample_ids in samples]
                            if data_utils.EOS_ID in sample:
                                sample = sample[:sample.index(data_utils.EOS_ID)]
                            ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in sample])
                            if grammar_critic.check_ans(sentence.split(), ans):
                                sample_correct += 1
                                if sentence.strip() in hist_samples:
                                    if ans not in hist_samples[sentence.strip()]:
                                        hist_samples[sentence.strip()].append(ans)
                                        coverage += 1.0 / grammar_critic.possible_ans_num(sentence.split())
                                else:
                                    hist_samples[sentence.strip()] = [ans]
                                    coverage += 1.0 / grammar_critic.possible_ans_num(sentence.split())
                            sample_count += 1
                print('argmax accuracy: {}'.format(float(argmax_correct) / argmax_count))
                print('sample accuracy: {}'.format(float(sample_correct) / sample_count))
                print('sample coverage: {}'.format(coverage / len(tmp_encoder_inputs)))
                print(len(tmp_encoder_inputs))
                for key, value in hist_samples.items():
                    fout.write("{}:{}\n".format(key, value))
                    
        elif FLAGS.test_type == 'print_test':
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            test_fout = os.path.join(FLAGS.model_dir, 'test_argmax_outs.txt')
            with open(test_data,'r') as f, open(test_fout,'w') as fout:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
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
                print(len(batch_encoder_inputs))

                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    for idx, sentence in enumerate(batch_sentences[batch_idx]):
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        fout.write(sentence)
                        fout.write(ans)
                        fout.write('\n')
                    
        elif FLAGS.test_type == 'perp':
            with open(FLAGS.test_data,'r') as f:
                all_f = f.readlines()
                batch_size = len(all_f)
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    # feature in my implementation
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)
                batch_encoder_inputs = []
                batch_encoder_lens = []
                encoder_inputs = []
                for idx in range(_buckets[-1][0]):
                    encoder_inputs.append(
                        np.array([tmp_encoder_inputs[batch_idx][idx]
                                  for batch_idx in range(batch_size)],
                                 dtype = np.int32))
                batch_encoder_inputs.append(encoder_inputs)
                batch_encoder_lens.append(tmp_encoder_lens)
                decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]

                total_perp = []
                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    for _ in range(10):
                        tmp_samples, log_probs = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                         decoder_inputs, mode='sample')
                        total_perp.extend(log_probs)
                    #outputs, perps = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                    #                                  decoder_inputs)
                print(float(sum(total_perp))/len(total_perp))
                
        elif FLAGS.test_type == 'KLD' or FLAGS.test_type == 'RKLD':
            grammar_critic = load_critic(FLAGS.test_critic)
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            out_name = 'KLD_hist_probs.txt'
            if 'R' in FLAGS.test_type:
                out_name = 'R'+out_name
            with open(test_data,'r') as f, open('KLD_hist_probs.txt','w') as fout:
                # gen input and reference output
                all_f = []
                all_ref = []
                if 'R' in FLAGS.test_type:
                    for _, line in enumerate(f):
                        all_ans_space = grammar_critic.get_ans_space()
                        for ans in all_ans_space:
                            all_f.append(line)
                            all_ref.append(ans)
                else:
                    for _, line in enumerate(f):
                        all_ans = grammar_critic.gen_ans(line)
                        for ans in all_ans:
                            all_f.append(line)
                            all_ref.append(ans)
                print(len(all_f))
                print(len(all_ref))
                # pack batch
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)
                tmp_decoder_inputs = []
                for _, ref in enumerate(all_ref):
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(ref), vocab, normalize_digits=False)
                    decoder_pad = [data_utils.EOS_ID] + [data_utils.PAD_ID] * (_buckets[-1][1] - len(token_ids)-2)
                    tmp_decoder_inputs.append([data_utils.GO_ID] + list(token_ids) + decoder_pad)

                batch_encoder_inputs = []
                batch_encoder_lens = []
                batch_decoder_inputs = []
                batch_sentences = []
                batch_ref = []
                batch_flag = 0
                for _ in range(int(len(tmp_encoder_inputs) / FLAGS.batch_size)):
                    encoder_inputs = []
                    for idx in range(_buckets[-1][0]):
                        encoder_inputs.append(
                            np.array([tmp_encoder_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(FLAGS.batch_size)],
                                     dtype = np.int32))
                    decoder_inputs = []
                    for idx in range(_buckets[-1][1]):
                        decoder_inputs.append(
                            np.array([tmp_decoder_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(FLAGS.batch_size)],
                                     dtype = np.int32))
                    batch_encoder_inputs.append(encoder_inputs)
                    batch_decoder_inputs.append(decoder_inputs)
                    batch_encoder_lens.append(tmp_encoder_lens[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_sentences.append(all_f[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_ref.append(all_ref[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_flag += FLAGS.batch_size

                print(len(batch_encoder_inputs))

                KLD, RKLD = 0, 0
                JSD1, JSD2 = 0, 0
                for batch_idx, (encoder_inputs, decoder_inputs) \
                        in enumerate(zip(batch_encoder_inputs, batch_decoder_inputs)):
                    outputs = model.stepwise_test_beam(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    outputs = outputs[0]
                    res_p = 1.0
                    current_sen = batch_sentences[batch_idx][0]
                    for idx, (sentence, ref) in enumerate(zip(batch_sentences[batch_idx], batch_ref[batch_idx])):
                        an_output = [output_ids[idx] for output_ids in outputs]
                        ref_tokens = data_utils.sentence_to_token_ids(tf.compat.as_bytes(ref), vocab, normalize_digits=False)
                        prob = 0
                        for gen_step in range(len(ref_tokens)):
                            try:
                                prob += np.log(an_output[gen_step][ref_tokens[gen_step]])
                            except:
                                print(gen_step)
                                print(ref_tokens)
                                print(ref)
                                print(len(an_output))
                                return

                        prob = np.exp(prob)
                        if sentence == current_sen:
                            res_p -= prob
                        else:
                            res_p = max(res_p, 1e-12)
                            RKLD += res_p * np.log(res_p / 1e-12)
                            JSD1 += res_p*np.log(2)
                            res_p = 1.0 - prob
                            current_sen = sentence
                        fout.write("{}//{}:{}\n".format(sentence, ref, prob))
                        if 'R' in FLAGS.test_type:
                            if grammar_critic.check_ans(sentence.split(), ref):
                                real_ref_prob = 1. / grammar_critic.possible_ans_num(sentence.split())
                            else:
                                real_ref_prob = 1e-12
                            KLD += real_ref_prob * np.log(real_ref_prob / (prob+1e-12))
                            RKLD += prob * np.log(prob / real_ref_prob)
                            JSD1 += prob*np.log(prob/(real_ref_prob+prob)*2)
                            JSD2 += real_ref_prob*np.log(real_ref_prob/(real_ref_prob+prob)*2)
                        else:
                            real_ref_prob = 1. / grammar_critic.possible_ans_num(sentence.split())
                            KLD += real_ref_prob * np.log(real_ref_prob / (prob+1e-12))

                print('KLD: {}'.format(KLD))
                print('RKLD: {}'.format(RKLD))
                print('JSD1: {}'.format(JSD1))
                print('JSD2: {}'.format(JSD2))
                    
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
                # feature in my implementation
                token_ids = list(reversed(token_ids)) + encoder_pad
                encoder_inputs = []
                for idx in token_ids:
                    encoder_inputs.append([idx])
                decoder_inputs = [[data_utils.GO_ID]]
                
                if FLAGS.test_type == 'realtime_argmax':
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
                    print(log_prob[0])

                elif FLAGS.test_type == 'realtime_beam_search':
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    for b in range(len(outs[0])):
                        outputs = [output_ids[b] for output_ids in outs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                        print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                        print(log_prob[b])

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
                        print(log_prob[b])
                        print(lm_log_prob[b])
                        print(lens[b])

                elif FLAGS.test_type == 'realtime_sample':
                    samples, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs, mode='sample')
                    sample = [sample_ids[0] for sample_ids in samples]
                    #if data_utils.EOS_ID in sample:
                    #    sample = sample[:sample.index(data_utils.EOS_ID)]
                    print(sample)
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in sample]))
                    print(log_prob)
                
                sys.stdout.write('> ')
                sys.stdout.flush()
                sentence = sys.stdin.readline()
