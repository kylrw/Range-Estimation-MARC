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

# Sets random seed to try and combat randomness in output
tf.random.set_seed(0)
np.random.seed(0)

# Loads matlab files data into a python dict
mat_data = loadmat('TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat')

#Test data set
test_data = loadmat('04_TEST_LGHG2@25degC_Norm_(05_Inputs).mat')

# Extracts the first 3 rows (Voltage, Current, Temp) and 100000 columns from the "X" key
x_train = mat_data["X"][:3,:100000] 
# Extracts the first 100000 columns from the "Y" (SOC) key
y_train = mat_data["Y"][:1,:100000] 

#Create the x_test and y_test data sets
x_test = test_data["X"][:3,:40000]
y_test = test_data["Y"][:1,:40000]


# Flips columns and rows so data is proper shape, have the same # of input features
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train),np.array(y_train)
x_test, y_test = np.array(x_test),np.array(y_test)

# Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# Define NN Network Architecture
# Using variables copied from MATLAB file

num_responses = 1

num_features = 3

num_hidden_units = 10

epochs = 100

learn_rate_drop_period = 1000

initial_learn_rate = 0.01

learn_rate_drop_factor = 0.01

batch_size = 10000

training_data_len = 80000

timesteps = 3

data_dim = 1

# Build the LSTM model
# Defined the model using same design as the Abstract
model = Sequential()
model.add(tf.keras.layers.Input(shape=(timesteps, data_dim),name="input"))
model.add(LSTM(num_hidden_units,name="LSTM"))
model.add(Dense(num_hidden_units,name="FullyConnectedLayer"))
model.add(tf.keras.layers.ReLU(max_value=1,name="ClippedRELU"))
model.add(Dense(1,name="output"))

# Compile the model, following MATLAB file again
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learn_rate),
  loss=tf.keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.MeanSquaredError()]
)

# Train the model
model.fit(
  x_train, y_train,
  epochs=epochs,
  batch_size=batch_size,
  validation_data=(x_train, y_train),
)

# Predicts SOC data using the trained LSTM model
#make predictions Stateful = True
trainPredict = model.predict(x_train, batch_size=batch_size)
# model.reset_states()
testPredict = model.predict(x_test, batch_size=batch_size)
# model.reset_states()

#Get the root mean squared error (RMSE)
rmse_train=np.sqrt(np.mean(((trainPredict- y_train)**2)))*100
print(rmse_train)

#Get the root mean squared error (RMSE)
rmse_test=np.sqrt(np.mean(((testPredict- y_test)**2)))*100
print(rmse_test)

# Plot the predictions
plt.plot(testPredict, label="Predictions")
# Plot the true values
plt.plot(y_test, label="Objective")
# Add a legend
plt.legend()
# Show the plot
plt.show()