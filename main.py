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
x_train = mat_data["X"][:3,:600000] 
# Extracts the first 100000 columns from the "Y" (SOC) key
y_train = mat_data["Y"][:1,:600000] 

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
batch_size = x_test.shape[0]
learn_rate_drop_period = 2000
LearningRate = 0.01
learn_rate_drop_factor = 0.5

# Build the LSTM model
# Defined the model using same design as the Abstract
model = Sequential()
model.add(LSTM(10, batch_input_shape=(batch_size, x_train.shape[1],1), stateful=True,return_sequences=False))
model.add(Dense(10))
model.add(tf.keras.layers.ReLU(max_value=1))
model.add(Dense(1))
model.summary()

#Define the learning rate scheduler
def scheduler(epoch, lr):
    if epoch % learn_rate_drop_period == 0 and epoch:
        return lr * learn_rate_drop_factor
    else:
        return lr

#Define the learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

#Compile the model with a learning rate scheduler
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LearningRate), loss='mean_squared_error')

#Train the model
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, callbacks=[lr_scheduler],validation_data=(x_test, y_test),)
# added validation data

# Predicts SOC data using the trained LSTM model
#make predictions Stateful = True
trainPredict = model.predict(x_train, batch_size=batch_size)
# model.reset_states()
testPredict = model.predict(x_test, batch_size=batch_size)
# model.reset_states()

#Get the root mean squared error (RMSE)
rmse_train=np.sqrt(np.mean(((trainPredict- y_train)**2)))*100
print("training data rmse", rmse_train)

#Get the root mean squared error (RMSE)
rmse_test=np.sqrt(np.mean(((testPredict- y_test)**2)))*100
print("test data rmse", rmse_test)

# Plot the predictions
plt.plot(testPredict, label="Predictions")
# Plot the true values
plt.plot(y_test, label="Objective")
# Add a legend
plt.legend()
# Show the plot
#plt.show()
