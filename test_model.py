'''
This program will take a already trained model in .tar.gz format and test it using tensorflow on a test dataset that is a .mat file.
And then display the reults in a graph using matplotlib.
'''

import scipy.io as sio
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle

# Load matlab file
mat_data = sio.loadmat('04_TEST_LGHG2@25degC_Norm_(05_Inputs).mat')

# concatenate the 'X' and 'Y' columns into one dataframe
df = pd.DataFrame(np.concatenate((mat_data['X'], mat_data['Y']), axis=0).T)

# removes the 3rd and 4th columns of the dataframe
df = df.drop([3, 4], axis=1)

# rename the columns of the dataframe
df.columns = ['V', 'I', 'T', 'SOC']

with open('Models/learner.pkl', 'rb') as f:
    model = pickle.load(f)

# convert dataframe to numpy array
test_data = df.to_numpy()

# make predictions using the model
predictions = model.predict(test_data)

# plot the predictions
plt.plot(predictions)

# plot the actual values
plt.plot(mat_data['Y'])

# show the plot
plt.show()

