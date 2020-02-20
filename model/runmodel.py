import tensorflow as tf 
from tensorflow import keras
import sys

base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/model/'

checkpoint_path = base_root + "cp.ckpt"

# load the weight
model = tf.keras.Model.load_weights(
  filepath=checkpoint_path,
  by_name=False,
  skip_mismatch=False
)
