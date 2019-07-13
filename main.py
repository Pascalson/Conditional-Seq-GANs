import re
import os
import sys

from args import *

from train_mle import *
from train_gan_n_rl import *
from test import *

if __name__ == '__main__':

    if FLAGS.test_type == 'None':
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
            for key, value in vars(FLAGS).items():
                f.write("{}={}\n".format(key, value))
        if FLAGS.gan_type == 'None' or FLAGS.gan_type == 'MLE':
            train_mle()
        else:
            train_gan()
    elif FLAGS.test_type == 'eval_mle':
        eval_mle()
    elif FLAGS.test_type == 'test_D' or FLAGS.test_type == 'batch_test_D':
        test_D()
    else:
        print(FLAGS.gan_type)
        test()
