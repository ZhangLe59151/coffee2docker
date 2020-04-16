# Create label for true ocr

import os

def get_ocr_label():
  path_true = '/Users/zhangle/Documents/IS/coffee2docker/data/ocr/label_true'
  path_false = '/Users/zhangle/Documents/IS/coffee2docker/data/ocr/label_false'
  
  files_true = os.listdir(path_true)
  files_false = os.listdir(path_false)
  
  s = {}
  for file in files_true:
    s[file] = 1
  for file in files_false:
    s[file] = 0
  
  print(s)
  return s