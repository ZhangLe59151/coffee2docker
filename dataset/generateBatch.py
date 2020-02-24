# -- Created by ZhangLe --#
# -- Created on 2020-02-18 --#

import tensorflow as tf
import os

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.enable_eager_execution()
sess = tf.compat.v1.Session()
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/dataset/'
base_root = '/Users/zhangle/Documents/IS/coffee2docker/dataset/'
filename = 'images.tfrecords'
# dataset = tf.data.TFRecordDataset(filenames = [filename])

num_workers = 1
worker_index = 0
num_epochs = 10
shuffle_buffer_size = 1
num_map_threads = 2
batch_size = 16

image_feature_description = {
  'height': tf.io.FixedLenFeature([], tf.int64),
  'width': tf.io.FixedLenFeature([], tf.int64),
  'depth': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def getShape(example_photo):
  parsed_features = tf.io.parse_single_example(example_photo, image_feature_description)
  height = parsed_features['height']
  width = parsed_features['width']
  return height, width

'''  
def parser_fn(example_photo):
  parsed_features = tf.io.parse_single_example(example_photo, image_feature_description)
  # labels = parsed_features['label']
  # images = parsed_features['image_raw']
  images = tf.image.decode_jpeg(parsed_features['image_raw'])
  images = tf.image.rgb_to_grayscale(images, name=None)
  heights = parsed_features['height']
  height = 4032
  widths = parsed_features['width']
  width = 3024
  image = tf.reshape(images, [height, width, 1])
  label = tf.cast(parsed_features['label'], tf.int64)
  return image

def parser_fn_label(example_photo):
  parsed_features = tf.io.parse_single_example(example_photo, image_feature_description)
  label = tf.cast(parsed_features['label'], tf.int64)
  return label
'''

d = tf.data.TFRecordDataset(base_root + filename)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
# dimage = d.map(parser_fn, num_parallel_calls=num_map_threads)
# dlabel = d.map(parser_fn_label, num_parallel_calls=num_map_threads)
# dimage = dimage.batch(batch_size, drop_remainder=True)

#trLabel = []
#trData = []
#for item in dimage:
  #print(item.numpy())
  #trData.append(item)


def parser_fn_all(example_photo):
  parsed_features = tf.io.parse_single_example(example_photo, image_feature_description)
  # labels = parsed_features['label']
  # images = parsed_features['image_raw']
  images = tf.image.decode_jpeg(parsed_features['image_raw'])
  heights = parsed_features['height']
  height = 1512
  widths = parsed_features['width']
  width = 1209
  print(images.shape)
  image = tf.reshape(images, [height, width, 1])
  label = tf.cast(parsed_features['label'], tf.int64)
  return image, label

d = d.map(parser_fn_all, num_parallel_calls=num_map_threads)
dataset = d.batch(batch_size, drop_remainder=False)
testdataset = d.batch(10, drop_remainder=False)