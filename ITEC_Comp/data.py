import pandas as pd
import os
import csv
import scipy.io as sio
import numpy as np

# In the folder OpenData within the current directory find every .mat file in the folders and subfolders of OpenData
# print the file name of all the .mat files found as well as the total number of files found

import os
import glob

# Get the current working directory
cwd = os.getcwd()

# Get the path to the OpenData folder
path = os.path.join(cwd, 'OpenData')

# Get the path to all the .mat files in the OpenData folder
mat_files = glob.glob(path + '/**/*.mat', recursive=True)

# Print the file name of all the .mat files found
for file in mat_files:
    print(file)

# Print the total number of .mat files found
print('Total number of .mat files found: ' + str(len(mat_files)))


filename = "train_data3.csv"

last_soc = 0

# Open the csv file and write the column names
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['SOC', 'V', 'I', 'P','V_avg_five','V_avg_one','I_avg'])

    soc = []
    v = []
    i = []
    v_avg_five = []
    v_avg_one = []
    i_avg = []
    p = []

    # Loop through all the mat files
    for file in mat_files:

        # Load the .mat file
        mat = sio.loadmat(file)
        # Get the ['meas']['Voltage'] data
        v = mat['meas']['Voltage'][0][0]
        # Get the ['meas']['Current'] data
        i = mat['meas']['Current'][0][0]
        # Get the ['meas']['SOC'] data
        soc = mat['meas']['SOC'][0][0]

        # ensure all data is not empty
        if len(soc) == 0 or len(v) == 0 or len(i) == 0:
            print("Empty data in file: " + file)
        # if the initial and final soc are within 1 percent of each other
        elif abs(soc[0] - soc[-1]) < 0.1:
            print("Initial and final SOC are within 1 percent of each other in file: " + file)
        else:
            # print the current file iteration and the name
            print("Processing file: " + file)

            #iterate through the lists and write the data to the csv file
            for x in range(len(soc)):

                # v_avg_five is the average voltage over the last 500 data points including the current data point
                v_avg_five.append(sum(v[max(0, x-499):x+1])/min(500, x+1))

                # v_avg_one is the average voltage over the last 100 data points including the current data point
                v_avg_one.append(sum(v[max(0, x-99):x+1])/min(100, x+1))

                # i_avg is the average current over the last 500 data points including the current data point
                i_avg.append(sum(i[max(0, x-499):x+1])/min(500, x+1))

                # p is the power
                p.append(v[x]*i[x])

                last_soc = soc[x][0]

                # Write the data to the csv file
                writer.writerow([soc[x][0], v[x][0], i[x][0], p[x][0], v_avg_five[x][0], v_avg_one[x][0], i_avg[x][0]])

#get the length of the csv file
num_lines = sum(1 for line in open(filename))

# send a text message when training is complete
import os
from twilio.rest import Client

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = "AC302209a332b7d9e3f441cdd0a5569ccf"
auth_token = "b467d59d57785221779f11f1400f3a37"
client = Client(account_sid, auth_token)
message = client.messages.create(
  body="Data Processing Complete! " + str(num_lines) + " lines of data processed.",
  from_="+13613457812",
  to="+13653661086"
)
print(message.sid)

