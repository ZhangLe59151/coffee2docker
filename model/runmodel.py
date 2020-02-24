import tensorflow as tf 
from tensorflow import keras
import sys
sys.path.append('../')
from dataset.generateBatch import dataset, testdataset

base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/model/'

checkpoint_path = base_root + "cp.ckpt"

# evaluate
filename = 'IMG_0336.jpg'
test_images = open('/Users/zhangle/Documents/IS/coffee2docker/dataset/testdata/' + filename, 'rb').read()
test_labels = 0

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(20, (5, 5), input_shape=(1512, 1209, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(40, (5, 5), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
  ])
  model.compile(optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])
  return model

model = create_model()
model.summary()

print('load weight path')
model.load_weights(checkpoint_path)

loss,acc = model.evaluate(testdataset, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))