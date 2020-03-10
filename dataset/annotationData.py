# -- Created by ZhangLe -- #
# -- Created on 2020-02-28 --#

import tensorflow as tf
import cv2
import os

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_root = '/Users/zhangle/Documents/IS/coffee2docker/dataset/'

BATCH_SIZE = 1
NUM_BOXES = 1
IMAGE_HEIGHT = 4032
IMAGE_WIDTH = 3024
CHANNELS = 3
CROP_SIZE = (24, 24)

image_string = open(base_root + 'data/IMG_0312.jpg', 'rb').read()
image = tf.image.decode_jpeg(image_string)

image = tf.random.normal(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) )
print(image.shape)
boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
print(output)