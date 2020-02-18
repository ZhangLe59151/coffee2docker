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
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

image_feature_description = {
  'height': tf.io.FixedLenFeature([], tf.int64),
  'width': tf.io.FixedLenFeature([], tf.int64),
  'depth': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))
  height = image_features['height']
  width = image_features['width']
  label = image_features['label']
  image = image_features['image_raw']