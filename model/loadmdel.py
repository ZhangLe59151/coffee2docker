# -- Created by ZhangLe --#
# -- Created on 2020-02-23 --#

import tensorflow as tf 
from tensorflow import keras

base_root = '/Users/zhangle/Documents/IS/coffee2docker/'
checkpoint_path = base_root + "/model/cp.ckpt"

new_model = tf.keras.models.load_model(checkpoint_path)
new_model.summery()