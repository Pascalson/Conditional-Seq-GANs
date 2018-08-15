import argparse
import re

def parse():
    parser = argparse.ArgumentParser(
        description='You have to set the parameters for seq2seq, \
        including both maximum likelihood estimation and \
        generative adversarial learning.')

    parser.add_argument("--pre-model-dir", type=str, default='None')
    parser.add_argument("--pre-D-model-dir", type=str, default='None')
    parser.add_argument("--model-dir", type=str, default='results/seq2seq')
    parser.add_argument("--data-dir", type=str, default='data/')
    parser.add_argument("--data-path", type=str, default='data/opensubtitle.txt')
    parser.add_argument("--steps-per-checkpoint", type=int, default=200)
    parser.add_argument("--fix-steps", type=int, default=0)
    # s2s: for encoder and decoder
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--grad-norm", type=float, default=5.0)
    parser.add_argument("--use-attn", type=bool, default=False)
    parser.add_argument("--vocab-size", type=int, default=14)
    parser.add_argument("--output-sample", type=bool, default=False)
    parser.add_argument("--input_embed", type=bool, default=True)
    # s2s: for training setting
    parser.add_argument("--buckets", type=str, default='[(10, 5)]')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--max-train-data-size", type=int, default=0)# 0: no limit
    # for value function
    parser.add_argument("--v-lr", type=float, default=1e-4)
    parser.add_argument("--v-lr-decay-factor", type=float, default=0.5)
    # gan: for discriminator
    parser.add_argument("--D-lr", type=float, default=1e-4)
    parser.add_argument("--D-lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--gan-type", type=str, default='None')
    parser.add_argument("--gan-size", type=int)
    parser.add_argument("--gan-num-layers", type=int)
    parser.add_argument("--G-step", type=int)
    parser.add_argument("--D-step", type=int)
    parser.add_argument("--option", type=str, default='None')
    # test
    parser.add_argument("--test-type", type=str, default='accuracy')
    parser.add_argument("--test-critic", type=str, default='None')
    parser.add_argument("--test-data", type=str, default='None')
    parser.add_argument("--test-fout", type=str, default='None')
    
    return parser.parse_args()

def parse_buckets(str_buck):
    _pair = re.compile(r"(\d+,\d+)")
    _num = re.compile(r"\d+")
    buck_list = _pair.findall(str_buck)
    if len(buck_list) < 1:
        raise ValueError("The bucket should has at least 1 component.")
    buckets = []
    for buck in buck_list:
        tmp = _num.findall(buck)
        d_tmp = (int(tmp[0]), int(tmp[1]))
        buckets.append(d_tmp)
    return buckets

FLAGS = parse()
FLAGS.data_dir, _ = FLAGS.data_path.rsplit('/',1)
_buckets = parse_buckets(FLAGS.buckets)
