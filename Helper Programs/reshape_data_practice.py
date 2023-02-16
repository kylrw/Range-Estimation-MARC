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
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        current_end = i + n_steps
        # check if we are beyond the dataset
        if current_end > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:current_end, :-1], sequences[current_end-1, -1]
       # seq_x, seq_y = [],[]
        #for j in range(i,current_end,1000):
        #    seq_x.append([sequences[j,0],sequences[j,1],sequences[j,2]])
        #seq_y = sequences[current_end-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

## prepare train data
# slices matlab data into each element
V = array(mat_data['X'][0,:])
I = array(mat_data['X'][1,:])
T = array(mat_data['X'][2,:])
SOC = array(mat_data['Y'][0,:])

# prepare individial elements for merge
V = V.reshape(len(V),1)
I = I.reshape(len(I),1)
T = T.reshape(len(T),1)
SOC = SOC.reshape(len(SOC),1)

# merge elements into one dataset
train_data = hstack((V,I,T,SOC))


#x_train, y_train = split_sequences(train_data, 10)
#print(x_train.shape, y_train.shape)

timesteps = 3

x_train, y_train = split_sequences(train_data, timesteps)

#print(x_train[0],y_train[0])

#print('Predictors matrix shape: ' + str(create_lstm_data(test_data, 10)[0].shape))
#print('Target array shape: ' + str(create_lstm_data(test_data, 10)[1].shape))




