#Import Libraries
import math
import numpy as np
import pandas
from scipy.io import loadmat
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Loads matlab files data into a python dict
mat_data = loadmat('SOCprediction\TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat')

# Extracts the first 3 rows (Voltage, Current, Temp) and 100000 columns from the "X" key
x_train = mat_data["X"][:3,:100000] 
# Extracts the first 100000 columns from the "Y" (SOC) key
y_train = mat_data["Y"][:1,:100000] 

x_train = pandas.DataFrame(x_train)

# Flips columns and rows so data is proper shape, have the same # of input features
x_train = x_train.T
y_train = y_train.T

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train),np.array(y_train)

# Define NN Network Architecture
# Using variables copied from MATLAB file

num_responses = 1

num_features = 3

num_hidden_units = 10

# 100 worked just as well if not better than 1000, im not sure why
# easier to debug so I left it at 100, but easily changeable
epochs = 100 

learn_rate_drop_period = 1000

initial_learn_rate = 0.01

learn_rate_drop_factor = 0.1

batch_size = 10000

training_data_len = 80000

# Build the LSTM model
# Defined the model using same design as MATLAB file
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(num_features,), dtype=tf.float32, name="input"),
  tf.keras.layers.BatchNormalization(center=True, scale=False, name="normalization"),
  tf.keras.layers.Dense(num_hidden_units, activation=None, name="fc1"),
  tf.keras.layers.Activation("tanh", name="tanh"),
  tf.keras.layers.Dense(num_hidden_units, activation=None, name="fc2"),
  tf.keras.layers.LeakyReLU(alpha=0.3, name="leaky_relu"),
  tf.keras.layers.Dense(num_responses, activation=None, name="fc3"),
  tf.keras.layers.ReLU(max_value=1, name="clipped_relu"),
  tf.keras.layers.Dense(1, activation=None, name="output")
])

# Compile the model, following MATLAB file again
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learn_rate),
  loss=tf.keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

# Train the model
model.fit(
  x_train, y_train,
  epochs=epochs,
  batch_size=batch_size,
  validation_data=(x_train, y_train),
)

# Regathers data to test
x_test = x_train#[training_data_len:]
y_test = y_train#[training_data_len:]

# Predicts SOC data using the trained LSTM model
y_pred = model.predict(x_test)

''' Tests accuracy for debugging, need to figure out proper algorithim
accuracy = 0
for i in range(y_pred.shape[0]):
  if y_pred[i] <= y_test[i]*0.01 or y_pred[i] >= y_test[i]*-0.01:
    accuracy += 1
accuracy = accuracy * 100 / y_pred.shape[0]
print("Accuracy = ", accuracy, "%")
'''

# Create final plot
plt.figure(figsize=(8,4))

# Plot the values
plt.plot(y_pred, label="Predictions")
plt.plot(y_test, label="Actual values")

plt.legend()
plt.show()