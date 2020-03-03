# -- Created by Zhang Le --#
# -- Created on 2020-03-01 --#

import cv2
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf

base_path = os.getcwd()
json_path = base_path + '/dataset/json/'
file_name2 = 'traindata.csv'
files = os.listdir(json_path)
'''
for item in files:
  with open(json_path + item, 'r') as f:
    temp = json.loads(f.read())
    width = temp['size']['width']
    height = temp['size']['height']
    label = temp['outputs']['object'][0]['name']
    xmin = temp['outputs']['object'][0]['bndbox']['xmin']
    ymin = temp['outputs']['object'][0]['bndbox']['ymin']
    xmax = temp['outputs']['object'][0]['bndbox']['xmax']
    ymax = temp['outputs']['object'][0]['bndbox']['ymax']
    path = temp['path']
    my_open = open(base_path + '/dataset/csv/traindata.csv', 'a')
    my_open.write(path + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + label + '\n')
    my_open.close()
'''

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def image_coco(
  image_string, 
  label, 
  height, 
  width, 
  num_objects, 
  filename, 
  xmin,
  xmax,
  ymin,
  ymax
  ):
  image_shape = tf.image.decode_jpeg(image_string).shape
  print(image_shape)
  feature = {
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'image/num_objects': _int64_feature(num_objects),
    #'image/filename': _bytes_feature(filename),
    #'image/source_id': _bytes_feature(filename),
    'image/encoded': _bytes_feature(image_string),
    #'image/format': _bytes_feature('jpeg'),
    'image/object/bbox/xmin': _float_list_feature(xmin),
    'image/object/bbox/xmax': _float_list_feature(xmax),
    'image/object/bbox/ymin': _float_list_feature(ymin),
    'image/object/bbox/ymax': _float_list_feature(ymax),
    'image/object/class/label': _int64_list_feature(label)
  }
  return tf.train.Example(features= tf.train.Features(feature = feature))


record_file = base_path + '/dataset/coco.tfrecords'
files = os.listdir(json_path)
with tf.io.TFRecordWriter(record_file) as writer:
  for item in files:
    with open(json_path + item, 'r') as f:
      temp = json.loads(f.read()) 
      if (temp['outputs']['object'][0]['name'] == "clothes"):
        label = 0
      else:
        label = 1
      xmin = temp['outputs']['object'][0]['bndbox']['xmin']
      ymin = temp['outputs']['object'][0]['bndbox']['ymin']
      xmax = temp['outputs']['object'][0]['bndbox']['xmax']
      ymax = temp['outputs']['object'][0]['bndbox']['ymax']
      path = temp['path']
      width = temp['size']['width']
      height = temp['size']['height']
      num_objects = len(temp['outputs']['object'])
      image_string = open(path, 'rb').read()
      tf_example = image_coco(
        image_string, 
        [label], 
        height, 
        width, 
        num_objects, 
        item, 
        [xmin],
        [xmax],
        [ymin],
        [ymax])
      writer.write(tf_example.SerializeToString())