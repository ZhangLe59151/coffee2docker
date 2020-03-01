import tensorflow as tf 
from tensorflow import keras
import sys
import cv2
sys.path.append('../')
from dataset.generateBatch import dataset, testdataset

base_root = '/Users/zhangle/Documents/IS/coffee2docker/'
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/'

checkpoint_path = base_root + "model/cp.ckpt"

# evaluate

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(20, (5, 5), input_shape=(806, 604, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
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

loss,acc = model.evaluate(dataset, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

print('model predict')
filename = 'IMG_0336.jpg'
# image = open(base_root + 'dataset/testdata/' + filename, 'rb').read()
pic = cv2.imread(base_root + 'dataset/testdata/' + filename, cv2.IMREAD_GRAYSCALE)
size = (806, 604) 
shrink = cv2.resize(pic, size, interpolation=cv2.INTER_AREA)
predict_data = tf.reshape(shrink, [1,806,604,1])

label = model.predict(
  predict_data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
  workers=1, use_multiprocessing=False
)

print(label)