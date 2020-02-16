import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import yolo3
sys.path.append('../')
from model import yolo3Model

def cv2plt(img):
  plt.axis('off')
  if np.size(img.shape) == 3:
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  else:
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)
  plt.show()

def pltDetect(img):
  (boxes, confidences) = yolo3Model.yoloV3Detect(img,scFactor=1/255,nrMean=(0,0,0),RBSwap=True, scoreThres=0.5,nmsThres=0.4)

  # lead the label
  classes  = open(lbl_file).read().strip().split("\n")  

  scoreThres      = 0.5
  nmsThres        = 0.4
  selected        = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=scoreThres, nms_threshold=nmsThres)

  # create color set
  colorset  = np.random.uniform(0,  255,  size=(len(classes),3)) 

  # extract color from detected class
  for j in selected[:,0]:
    box     = boxes[j]
    color   = colorset[classId[j]]
    txtlbl  = str(classes[classId[j]])
    x       = int(box[0])
    y       = int(box[1])
    w       = int(box[2])
    h       = int(box[3])
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv2.putText(img, txtlbl, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
  cv2plt(img)

