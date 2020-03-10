#  -- Created by Zhang Le --#
#  -- Created on 2020-03-01 -- #
import cv2
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
import random
import logging

from component import datasetUtil

base_path = os.getcwd()
base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'

# read label dict
label_map_dict = {
  'chair': 1,
  'sofa': 2,
  'table': 3
}

image_dir = base_path + '/data/images'
annotations_dir = base_path + '/data/annotations'
example_list = datasetUtil.read_images_list(image_dir)


def create_tf_record(output_filename,label_map_dict,
  annotations_dir,image_dir,examples,faces_only=True,mask_type='png'):
  for item in train_examples:
    data = datasetUtil.read_json(annotations_dir + item.replace('.jpg','.json'))
    tf_example = dict_to_tf_example(
      data,
      mask_path,
      label_map_dict,
      image_dir,
      faces_only=faces_only,
      mask_type=mask_type)
  with tf.io.TFRecordWriter(output_filename) as output_tfrecords:
    output_tfrecords.write(tf_example.SerializeToString())


# proform data
random.seed(42)
random.shuffle(example_list)
num_examples = len(example_list)
num_train = int(0.7 * num_examples)
train_examples = example_list[:num_train]
val_examples = example_list[num_train:]
logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
train_output_path = base_path + '/data/tffile/train.tfrecord'
val_output_path = base_path + '/data/tffile/val.tfrecord'
create_tf_record(
      train_output_path,
      label_map_dict,
      annotations_dir,
      image_dir,
      train_examples)

