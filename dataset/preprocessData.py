# -- Created by ZhangLe --#
# -- Created on 2020-02-20 --#

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

BATCH_SIZE = 1
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
CROP_SIZE = (24, 24)

image_string = open('./data/IMG_0312.jpg', 'rb').read()
image = tf.image.decode_jpeg(image_string)
boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
print(boxes)
box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
print(box_indices)
output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
#
# 
# 
# output.shape  #=> (5, 24, 24, 3)