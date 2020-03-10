import cv2
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
import random
import logging

from component import datasetUtil

base_path = os.getcwd()

filename = '/Users/zhangle/Documents/IS/T3CA1/annotations/instances_train2014.json'
data = datasetUtil.read_json(filename)
print(data['images'][1])

images = []
filenamelist = datasetUtil.read_images_list(base_path+'/data/image')
for item in filenamelist:
  image = {
    'license': 1,
    'file_name': item,
    'height': 100,
    'width': 200,
    'id': 0 }
  images.append(image)



cocofile = {
  'info': {},
  'licenses': [],
  'images': images,
  'annotations': [],
  'categories': []
}

print(cocofile)