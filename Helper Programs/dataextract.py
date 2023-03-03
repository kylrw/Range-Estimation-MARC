'''
This program extracts the data from multiple .mat files in the same folder (Data/1)
Then concatenates them and saves them as a .csv file.

'''

import scipy.io as sio
import numpy as np
import os
import pandas as pd

# Get the current working directory
cwd = os.getcwd()

# Get the list of files in the folder
files = os.listdir(cwd + '/Data/1')

# Create an empty list to store the data
data = []

# Loop through the files
for file in files:
    # Load the data
    data.append(sio.loadmat(cwd + '/Data/1/' + file))

# Creates empty data frames
X = []
Y = []

# Loop through the data
for i in range(len(data)):
    # Get the data
    X.append(data[i]['X'])
    Y.append(data[i]['Y'])

# Appends each column into a single column
X = np.concatenate(X, axis=1)
Y = np.concatenate(Y, axis=1)

# Print the shape of the data
print(X.shape)
print(Y.shape)

#Transposes the data
X = X.T
Y = Y.T

#Appends Y to X in the initial column
X = np.append(Y, X, axis=1)

#Labels each column in X: SOC, V, I, T, V_0.5mHz, I_0.5mHz, V_5mHz, I_5mHz
X = pd.DataFrame(X, columns=['SOC','V', 'I', 'T', 'V_0.5mHz', 'I_0.5mHz', 'V_5mHz', 'I_5mHz'])
#Converts X into a csv file
df = pd.DataFrame(X)
df.to_csv('X.csv', index=False)

