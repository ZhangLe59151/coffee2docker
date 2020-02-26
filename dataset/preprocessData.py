# -- Created by ZhangLe --#
# -- Created on 2020-02-20 --#

import tensorflow as tf
import cv2
import os

tf.compat.v1.enable_eager_execution()

base_root = '/Users/zhangle/Documents/IS/coffee2docker/dataset/'

'''
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
'''

def Corpimg(filename):
  pic = cv2.imread(base_root + 'data/' + filename, cv2.IMREAD_GRAYSCALE)
  height = pic.shape[0]
  width = pic.shape[1]
  size = (int(width * 0.2), int(height*0.2))  
  print(size)
  shrink = cv2.resize(pic, size, interpolation=cv2.INTER_AREA)
  cv2.imwrite(base_root + 'traindata/'+filename, shrink)

files = os.listdir(base_root + 'data/')
for file in files:
  Corpimg(file)

