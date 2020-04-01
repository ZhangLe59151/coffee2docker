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
# print(data['annotations'][0]['category_id'])

images = []
filenamelist = datasetUtil.read_images_list(base_path+'/data/annotation')

data = datasetUtil.read_json(base_path + '/data/annotation/image_0.json')
# print(data['size']['width'])

annotations = []
for item in filenamelist:
  data = datasetUtil.read_json(base_path + '/data/annotation/' + item)
  # polygon = data['outputs']['object'][0]['polygon']
  for i in data['outputs']['object']:
    id = 63
    if (i['name']) == 'sofa':
      id = 63
    if (i['name']) == 'table':
      id = 67
    if (i['name']) == 'chair':
      id = 62
    polygon = []
    for j in i['polygon'].values():
      polygon.append(j)
    annotation = {
      'segmentation': [polygon],
      'category_id': id,
      'image_id': item.replace('.json','.jpg')
    }
    annotations.append(annotation)

  image = {
    'license': 1,
    'file_name': item.replace('.json','.jpg'),
    'height': data['size']['height'],
    'width': data['size']['width'],
    'id': item.replace('.json','.jpg') }
  images.append(image)



cocofile = {
  'info': { "version": "1.0" },
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


filejson = base_path + '/data/coco.json'
coco = json.dumps(cocofile)
with open(filejson, 'w') as f:
  f.write(coco)
