# -- Created by ZhangLe --#
# -- Created on 2020-02-19 --#

import tensorflow as tf 
from tensorflow import keras
import sys
sys.path.append('../')
from dataset.generateBatch import dataset


base_root = '/Users/zhangle/Documents/IS/coffee2docker/model/'
# base_root = '/Users/zhangle/Documents/TableDetect/coffee2docker/model/'

inputs = tf.keras.Input(shape=(4032, 3024, 1))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(20, (5, 5), input_shape=(4032, 3024, 1), activation='relu'),
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

checkpoint_path = base_root + "cp.ckpt"

model.save_weights(checkpoint_path.format(epoch=0))

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
  save_weights_only=False, 
  verbose=1)

# Train the model with the new callback
model.fit(x=dataset, y=None, epochs=10,
  validation_data=dataset,
  callbacks=[cp_callback])  # Pass callback to training

# save the model
Model_path = base_root + '/checkpoint'
model.save(
  filepath = Model_path,
  overwrite=True,
  include_optimizer=True,
  save_format=None,
  signatures=None,
  options=None
)
