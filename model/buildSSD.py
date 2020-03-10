# -- Created by ZhangLe --#
# -- Created on 2020-03-04 --#

import tensorflow as tf 
from tensorflow import keras
import sys,os
import numpy as np

base_path = os.getcwd()

'''
tf.nn.conv2d(
  input, filters, strides, padding, data_format='NHWC', dilations=None, name=None
)
'''

input_data = tf.Variable( np.random.rand(2,4,4,2), dtype = np.float32 )
filter_data = tf.Variable( np.random.rand(4, 4, 2, 3), dtype = np.float32)
y = tf.nn.conv2d(
  input_data, 
  filter_data, 
  strides = [1, 1, 1, 1], 
  padding = 'SAME')
print(y)
