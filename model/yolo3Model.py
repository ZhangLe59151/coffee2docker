import sys, os
import cv2
import numpy as np

def getOutputLayers(net):
  layers    = net.getLayerNames()
  outLayers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  return outLayers

def yoloV3Detect(img, scFactor=1/255, nrMean=(0,0,0), RBSwap=True, scoreThres=0.5, nmsThres=0.4):
  lbl_file = '../model/yolov3.txt'
  imgHeight = img.shape[0]
  imgWidth  = img.shape[1] 

  # lead the label
  print(lbl_file)
  classes  = open('../model/yolov3.txt').read().strip().split("\n")  

  # load the net model
  yoloconfig = '../model/yolov3.cfg'
  yoloweights= '../model/yolov3.weights'
  net        = cv2.dnn.readNet(yoloweights, yoloconfig)

  # create blob
  blob  = cv2.dnn.blobFromImage(image=img, scalefactor=1/255, size=(416, 416), mean=(0,0,0), swapRB=True, crop=False)

  # out put 3 layers
  net.setInput(blob)
  outLyrs  = getOutputLayers(net)
  preds    = net.forward(outLyrs)

  # 
  classId         = [] # class
  confidences     = [] # confidence level
  fboxes          = [] # position and the size of the box

  # extract the result form preds
  for scale in preds:
    for pred in scale:
      scores      = pred[5:]
      clss        = np.argmax(scores)
      confidence  = scores[clss]

      if confidence > 0.5:
        xc      = int(pred[0]*imgWidth) # box center
        yc      = int(pred[1]*imgHeight) # box center
        w       = int(pred[2]*imgWidth) # box width
        h       = int(pred[3]*imgHeight)
        x       = xc - w/2
        y       = yc - h/2
        classId.append(clss)
        confidences.append(float(confidence))
        fboxes.append([x, y, w, h]) 
  fclasses = confidences
  return [fboxes,fclasses]