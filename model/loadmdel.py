# -- Created by ZhangLe --#
# -- Created on 2020-02-23 --#

import tensorflow as tf 
from tensorflow import keras

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'
checkpoint_path = base_root + "cp.ckpt"

print(checkpoint_path)
