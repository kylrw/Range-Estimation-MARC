#Import Libraries
import math
import numpy as np
from numpy import array, hstack
import pandas
from scipy.io import loadmat
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Loads matlab files data into a python dict
mat_data = loadmat('TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat')

#Test data set
mat_data_2 = loadmat('04_TEST_LGHG2@25degC_Norm_(05_Inputs).mat')


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, Y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        current_end = i + n_steps
        # check if we are beyond the dataset
        if current_end > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:current_end, :-1], sequences[current_end-1, -1]
        X.append(seq_x)
        Y.append(seq_y)
    return array(X), array(Y)

## Define NN Network Architecture
# Using variables copied from MATLAB file

num_responses = 1
num_features = 3
num_hidden_units = 10
epochs = 1000
batch_size = 100
learn_rate_drop_period = 2000
LearningRate = 0.01
learn_rate_drop_factor = 0.5
timesteps = 5
n_features = 3

## prepare train data
# slices matlab data into each element
V = array(mat_data['X'][0,:100000])
I = array(mat_data['X'][1,:100000])
T = array(mat_data['X'][2,:100000])
SOC = array(mat_data['Y'][0,:100000])

# prepare individial elements for merge
V = V.reshape(len(V),1)
I = I.reshape(len(I),1)
T = T.reshape(len(T),1)
SOC = SOC.reshape(len(SOC),1)

# merge elements into one dataset
train_data = hstack((V,I,T,SOC))

## prepare test data
# slices matlab data into each element
V = array(mat_data_2['X'][0,:40000])
I = array(mat_data_2['X'][1,:40000])
T = array(mat_data_2['X'][2,:40000])
SOC = array(mat_data_2['Y'][0,:40000])

# prepare individial elements for merge
V = V.reshape(len(V),1)
I = I.reshape(len(I),1)
T = T.reshape(len(T),1)
SOC = SOC.reshape(len(SOC),1)

# merge elements into one dataset
test_data = hstack((V,I,T,SOC))


x_train, y_train = split_sequences(train_data, timesteps)
print(x_train.shape, y_train.shape)

x_test, y_test = split_sequences(test_data, timesteps)
print(x_test.shape, y_test.shape)

## Build the LSTM model
# Defined the model using same design as the Abstract
model = Sequential()
#model.add(LSTM(10, batch_input_shape=(0,x_train.shape[1],n_features), stateful=True,return_sequences=False))
model.add(LSTM(10, activation='relu', input_shape=(timesteps, n_features)))
model.add(Dense(1))
model.add(tf.keras.layers.ReLU(max_value=1))
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
model.fit(x_train,y_train, epochs=epochs, callbacks=[lr_scheduler],validation_data=(x_test, y_test),)

## Predicts SOC data using the trained LSTM model
# make predictions Stateful = True
trainPredict = model.predict(x_train, batch_size=batch_size)
testPredict = model.predict(x_test, batch_size=batch_size)

#Get the root mean squared error (RMSE)
rmse_train=np.sqrt(np.mean(((trainPredict[:1000] - y_train[:1000])**2)))*100
print("training data rmse", rmse_train)

#Get the root mean squared error (RMSE)
rmse_test=np.sqrt(np.mean(((testPredict[:1000] - y_test[:1000])**2)))*100
print("test data rmse", rmse_test)

# Plot the predictions
plt.plot(testPredict, label="Predictions")
# Plot the true values
plt.plot(y_test, label="Objective")
# Add a legend
plt.legend()
# Show the plot
plt.savefig('prediction.png')
#plt.show()
