import tensorflow as tf 
from tensorflow import keras
import sys
from buildmodel import create_model

base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/model/'

checkpoint_path = base_root + "cp.ckpt"

model = create_model()
model.load_weights(checkpoint_path)

# evaluate
filename = 'IMG_0336.jpg'
test_images = open('datatest/' + filename, 'rb').read()
test_labels = [0]
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))