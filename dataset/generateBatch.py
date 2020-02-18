# -- Created by ZhangLe --#
# -- Created on 2020-02-18 --#

import tensorflow as tf 
from tfrecordCreate import image_feature_description

filename = 'images.tfrecords'
# dataset = tf.data.TFRecordDataset(filenames = [filename])

num_workers = 1
worker_index = 0
num_epochs = 1
shuffle_buffer_size = 1
num_map_threads = 1

def parser_fn(example_photo):
  features = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
  }
  parsed_features = tf.parse_single_example(example_photo, features=features)
  labels = parsed_features['label']
  images = parsed_features['image_raw']
  return images, labels

d = tf.data.TFRecordDataset(filename)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
dataset = d.map(parser_fn, num_parallel_calls=num_map_threads)

for imgs, labels in dataset:
  print(imgs)
