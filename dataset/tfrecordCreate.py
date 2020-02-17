# -- Create by Zhang Le --#
# -- Create on 2020-02-17 --#

import tensorflow as tf
import numpy as np
from PIL import Image
import os

tf.compat.v1.enable_eager_execution()
cwd = os.getcwd()
root = cwd

# label include filename and label
image_labels = {
  'IMG_0317.png': 0,
}

image_string = tf.io.read_file(cwd + '/IMG_0317.png')
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
  print(image_shape)
  feature = {
    'height': _int64_feature(image_shape[0]),
    'width': _int64_feature(image_shape[1]),
    'depth': _int64_feature(image_shape[2]),
    'label': _int64_feature(label),
    'image_raw': _bytes_feature(image_string),
  }
  return tf.train.Example(features= tf.train.Features(feature = feature))

'''
# Print the Example Message to test
for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')
'''

record_file = 'IMG_0317.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())