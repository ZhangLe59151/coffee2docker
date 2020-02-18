# -- Created by ZhangLe --#
# -- Created on 2020-02-18 --#

import tensorflow as tf 
from tfrecordCreate import image_feature_description

#tf.compat.v1.enable_eager_execution()
sess = tf.compat.v1.Session()
filename = 'images.tfrecords'
# dataset = tf.data.TFRecordDataset(filenames = [filename])

num_workers = 1
worker_index = 0
num_epochs = 1
shuffle_buffer_size = 1
num_map_threads = 2

def getShape(example_photo):
  parsed_features = tf.io.parse_single_example(example_photo, image_feature_description)
  height = parsed_features['height']
  width = parsed_features['width']
  return height, width
  
def parser_fn(example_photo):
  parsed_features = tf.io.parse_single_example(example_photo, image_feature_description)
  # labels = parsed_features['label']
  # images = parsed_features['image_raw']
  images = tf.image.decode_jpeg(parsed_features['image_raw'])
  heights = parsed_features['height']
  height = 4032
  widths = parsed_features['width']
  width = 3024
  image = tf.reshape(images, [height, width, 3])
  label = tf.cast(parsed_features['label'], tf.int64)
  return image, label

d = tf.data.TFRecordDataset(filename)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
dataset = d.map(parser_fn, num_parallel_calls=num_map_threads)

trLabel = []
trData = []
for item, label in dataset:
  # print(item.numpy())
  trData.append(item.numpy())
  trLabel.append(label.numpy())


