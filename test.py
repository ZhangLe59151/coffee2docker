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
print(data['annotations'][0]['category_id'])
print(data['images'][0]['width'])
print(data['images'][0]['height'])

images = []
filenamelist = datasetUtil.read_images_list(base_path+'/data/annotation')

data = datasetUtil.read_json(base_path + '/data/annotation/image_0.json')
print(data['outputs']['object'][0]['name'])

annotations = []
for item in filenamelist:
  data = datasetUtil.read_json(base_path + '/data/annotation/' + item)
  polygon = data['outputs']['object'][0]['polygon'].values()
  for i in data['outputs']['object']:
    id = 63
    if (i['name']) == 'sofa':
      id = 63
    if (i['name']) == 'table':
      id = 67
    if (i['name']) == 'chair':
      id = 62
    annotation = {
      'segmentation': [i['polygon'].values()],
      'category_id': id,
      'image_id': item.replace('.json','.jpg')
    }
    annotations.append(annotation)
  
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
  'annotations': annotations,
  'categories': [
    {"supercategory": "furniture", "id": 62, "name": "chair"},
    {"supercategory": "furniture", "id": 63, "name": "couch"},
    {"supercategory": "furniture", "id": 65, "name": "bed"},
    {"supercategory": "furniture", "id": 67, "name": "dining table"}
  ]
}

print(cocofile)
