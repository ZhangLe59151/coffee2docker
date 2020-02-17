# -- created by ZhangLe --#
# -- created on 2020-02-17 --#

import tensorflow as tf
import numpy as np
from PIL import Image
import IPython.display as display
import os
from matplotlib import pyplot as plt

tf.compat.v1.enable_eager_execution()
cwd = os.getcwd()
root = cwd
raw_image_dataset = tf.data.TFRecordDataset('IMG_0317.tfrecords')

image_labels = {
  'clothes': 0,
  'pants': 1,
}

image_string = open(cwd + "/IMG_0317.png", 'rb').read()
label = 0

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

image_feature_description = {
  'height': tf.io.FixedLenFeature([], tf.int64),
  'width': tf.io.FixedLenFeature([], tf.int64),
  'depth': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def image_example(image_string, label):
  image_shape = tf.image.decode_png(image_string).shape
  feature = {
    'height': _int64_feature(image_shape[0]),
    'width': _int64_feature(image_shape[1]),
    'depth': _int64_feature(image_shape[2]),
    'label': _int64_feature(label),
    'image_raw': _bytes_feature(image_string),
  }
  return tf.train.Example(features= tf.train.Features(feature = feature))

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))
  height = image_features['height']
  width = image_features['width']