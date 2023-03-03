"""
This program will train a FNN to predict the range of an Electric Vehicle using tensorflow

The training data has the following columns, in order:
    - File Name
    - Initial SOC
    - Final SOC
    - Initial Altitude
    - Final Altitude
    - Average Speed
    - Accumulated Distance

The target is the Final SOC

The program will use the following data to train the FNN model, to predict the Final SOC:
    - Initial SOC
    - Initial Altitude
    - Final Altitude
    - Average Speed
    - Accumulated Distance

To test the model, the program will recieve the following data from the user:
    - Initial SOC
    - Initial Altitude
    - Final Altitude
    - Average Speed
    - Accumulated Distance
"""

# Importing the libraries
import math
import numpy as np
from numpy import array, hstack
import pandas as pd
#pipimport sklearn
from scipy.io import loadmat
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

# Importing the dataset
dataset = pd.read_csv('Data/RangeDataReformated.csv')

# Splitting the dataset into the Training set and Test set
# Add the 2nd column to X
X = dataset.iloc[:, 1].values
# Add the 4th to 7th columns to X
X = np.column_stack((X, dataset.iloc[:, 3:7].values))

# Add the 3rd column to y
y = dataset.iloc[:, 2].values

# Normalizing the data
# Normalize the X data
X = X / X.max(axis=0)
# Normalize the y data
y = y / y.max(axis=0)

# Build the FNN model
model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, epochs=1000, batch_size=10)

# Predict the range
# Get the user input
initialSOC = float(input("Enter the initial SOC: "))
initialAltitude = float(input("Enter the initial altitude: "))
finalAltitude = float(input("Enter the final altitude: "))
averageSpeed = float(input("Enter the average speed: "))
accumulatedDistance = float(input("Enter the accumulated distance: "))
# Normalize the user input
initialSOC = initialSOC / 100
initialAltitude = initialAltitude / 1000
finalAltitude = finalAltitude / 1000
averageSpeed = averageSpeed / 100
accumulatedDistance = accumulatedDistance / 1000
# Predict the range
predictedRange = model.predict([[initialSOC, initialAltitude, finalAltitude, averageSpeed, accumulatedDistance]])

# Print the predicted range
print("The predicted range is: ", predictedRange)

# Save the model
model.save('Models/range.h5')





