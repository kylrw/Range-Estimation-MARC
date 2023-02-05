"""
This program file will take a .mat file and convert it into a python dictionary.
It will then output the python dicitonary into a .csv file.
"""

import scipy.io as sio
import csv
import numpy as np
import pandas as pd

# Load matlab file
mat_data = sio.loadmat('TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat')

# Print all column names of the matlab file
print(mat_data.keys())

# concatenate the 'X' and 'Y' columns into one dataframe
df = pd.DataFrame(np.concatenate((mat_data['X'], mat_data['Y']), axis=0).T)

# removes the 3rd and 4th columns of the dataframe
df = df.drop([3, 4], axis=1)

# adds a datetime column to the dataframe in the first column with the format YYYY/MM/DD HH:MM:SS
#df.insert(0, 'Date', pd.date_range('1/1/2019', periods=len(df), freq='1min'))

#adds an index column to the dataframe in the first column
#df.insert(0, 'index', range(0, len(df)))

# Print the first 5 rows of the dataframe
print(df.head())

# Rename the columns of the dataframe
df.columns = ['V', 'I', 'T', 'SOC']

# Print the first 5 rows of the dataframe
print(df.head())

#convert dataframe to csv
df.to_csv('SOC_TrainData.csv', index=False)

