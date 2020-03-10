# -- Created on 2020-03-09 --#

import tensorflow as tf
import os
import json

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_images_list(path):
  imagesList = []
  files= os.listdir(path)
  for file in files:
    imagesList.append(file)
  return imagesList

def read_json(path):
  temp = ''
  with open(path, 'r') as f:
    temp = json.loads(f.read())
  return temp