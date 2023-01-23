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



#data = mat_data[['Voltage','Current','Temp','V_avg','I_avg']]

def create_lstm_data(data, k):
    '''
    input: 
        data - numpy matrix of shape (n, p+1)
        k - the length of the sequence of previous frames required to predcit the next value Y

    output:
        x_data - the predictors numpy matrix of shape (n-k, k, p)
        y_data - the target numpy matrix of shape (n-k, 1)
    '''

    x_data = np.zeros([data.shape[0]-k, k, data.shape[1]-1])
    y_data = []

    for i in range(data.shape[0]):
        current_sequence = data[i-k: i, :-1]
        current_target = data[i-1, -1]

        x_data[i-k,:,:] = current_sequence.reshape(1,k,x_data.shape[2])
        y_data.append(current_target)

    return x_data, np.asarray(y_data)

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
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

## prepare test data
# slices matlab data into each element
V = array(mat_data_2['X'][0,:])
I = array(mat_data_2['X'][1,:])
T = array(mat_data_2['X'][2,:])
SOC = array(mat_data_2['Y'][0,:])

# prepare individial elements for merge
V = V.reshape(len(V),1)
I = I.reshape(len(I),1)
T = T.reshape(len(T),1)
SOC = SOC.reshape(len(SOC),1)

# merge elements into one dataset
test_data = hstack((V,I,T,SOC))


x_train, y_train = split_sequences(train_data, 10)
print(x_train.shape, y_train.shape)

x_test, y_test = split_sequences(test_data, 10)
print(x_test.shape, y_test.shape)

# Convert the x_train and y_train to numpy arrays
#train_data = np.array[train_data]



#print('Predictors matrix shape: ' + str(create_lstm_data(data, 10)[0].shape))



#print('Predictors matrix shape: ' + str(create_lstm_data(data.values, 10)[0].shape))
#print('Target array shape: ' + str(create_lstm_data(data.values, 10)[1].shape))
