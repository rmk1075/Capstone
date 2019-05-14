#load the saved model and evaluate
import tensorflow as tf
import numpy as np
import os

import main_train

from tensorflow import keras
from tensorflow.keras import layers

#training set
dirname = '/path/to/the/project/dir'
data = np.load(dirname + '/path/to/the/data.npz')
x_data = data['X']
y_data = data['Y']

learning_rate = 0.001
training_epochs = 1000
batch_size = 100

dir_name = "/path/to/the/dir"

#load trained model
model = tf.contrib.saved_model.load_keras_model(dir_name + "/path/to/the/model/dir/")
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.evaluate(x_data, y_data)

#load latest checkpoint
latest = tf.train.latest_checkpoint(dir_name)

new_model = main_train.create_model()
new_model.load_weights(latest)

new_model.evaluate(x_data, y_data)
