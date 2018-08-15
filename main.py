import re
import os
import sys

from args import *

####################
# In Beta
#from replay_memory import *
####################

from train_mle import *
from train_gan_n_rl import *
from train_others import *
from test import *
from test_others import *

if __name__ == '__main__':

    if FLAGS.test_type == 'None':
        if FLAGS.option == 'pretrain_D':
            train_critic()
        elif FLAGS.option == 'pretrain_V':
            train_value()
        else:
            if not os.path.exists(FLAGS.model_dir):
                os.makedirs(FLAGS.model_dir)
            with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
                for key, value in vars(FLAGS).items():
                    f.write("{}={}\n".format(key, value))
            if FLAGS.gan_type == 'None':
                train_mle()
            else:
                train_gan()
    elif FLAGS.test_type == 'test_D' or FLAGS.test_type == 'batch_test_D':
        test_D()
    else:
        test()
