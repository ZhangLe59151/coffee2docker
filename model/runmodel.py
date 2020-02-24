import tensorflow as tf 
from tensorflow import keras
import sys

base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/model/'

checkpoint_path = base_root + "cp.ckpt"

# evaluate
filename = 'IMG_0336.jpg'
test_images = open('/Users/zhangle/Documents/IS/coffee2docker/dataset/testdata/' + filename, 'rb').read()
test_labels = 0


model = tf.keras.models.load_model(checkpoint_path)
model.summery()

print('load weight path')
model.load_weights(checkpoint_path)

loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))