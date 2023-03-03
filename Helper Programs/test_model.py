'''
This program reformats the training data to take one row for every 10 rows of the original data


'''

# Importing the libraries
import math
import numpy as np
from numpy import array, hstack
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data/SOC_TrainData.csv')

#plot the original data, SOC column
plt.plot(dataset.iloc[:, 3])
plt.show()

# Filter out the data to take one row for every 10 rows of the original data
dataset = dataset.iloc[::10, :]
# Reset the index
dataset = dataset.reset_index(drop=True)

# Output into a new csv file
dataset.to_csv('Data/trial.csv', index=False)

# plot the resulting data, SOC column
plt.plot(dataset.iloc[:, 3])
plt.show()