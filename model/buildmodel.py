# -- Created by ZhangLe --#
# -- Created on 2020-02-19 --#

import tensorflow as tf 
import sys
sys.path.append('../')
from dataset.generateBatch import trData, trLabel

inputs = tf.keras.Input(shape=(4032, 3024, 3))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.fit(
  x=trData,
  y=trLabel,
  batch_size=None,
  epochs=1,
  verbose=1,
  callbacks=None,
  validation_split=0.0,
  validation_data=None,
  shuffle=True,
  class_weight=None,
  sample_weight=None,
  initial_epoch=0,
  steps_per_epoch=None,
  validation_steps=None,
  validation_freq=1,
  max_queue_size=10,
  workers=1,
  use_multiprocessing=False)

print(model)
