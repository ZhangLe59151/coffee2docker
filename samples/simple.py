# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import readIMG
sys.path.append('../')
from component import readIMG

## testing
import cv2

imgdir = '/Users/zhangle/Documents/Clothes/IMG_0312.png'
img = cv2.imread(imgdir)
# readIMG.cv2plt(img)
readIMG.pltDetect(img)