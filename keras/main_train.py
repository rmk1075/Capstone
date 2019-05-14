#새로운 model을 만드는 NN, test accuracy는 생략 - 전체 feature에 대한 training accuracy만 측정
#_*_coding: utf-8_*_
import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers

#training set
dirname = '/path/to/the/project/dir'
data = np.load(dirname + '/path/to/the/data.npz')
x_data = data['X']
y_data = data['Y']

learning_rate = 0.001
training_epochs = 10000000
batch_size = 100

#create neural net
def create_model():
  model = tf.keras.Sequential([
      layers.Dense(300, activation=keras.activations.sigmoid),
      layers.Dense(200, activation=keras.activations.relu),
      layers.Dense(100, activation=keras.activations.relu),
      layers.Dropout(0,2),
      layers.Dense(9, activation=keras.activations.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
  #oprimizer: training process?
  #loss: loss function - which be minimalized by the minimalizing process
  #metrics: the object for monitoring the training - tf.keras.metrics

  return model

#checkpoint
checkpoint_dir = "/path/to/the/checkpoint"
if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)

#checkpoint - period  100
checkpoint_path = checkpoint_dir + "cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=100)

model = create_model()

#train model
model.fit(x_data, y_data, epochs=1000, batch_size=batch_size, callbacks=[cp_callback])
