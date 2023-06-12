"""
This program will read all xlsx files, search all sheets for the columns:

    - SOC
    - Voltage
    - Current
    - Temperature

When a columns has been found, it will be added to the respective list.
If a column is not found any where in the file, the file will be skipped, and a message will be printed to the console.
If all columns are successfully found, they will be added to a csv file, until the csv file has >= 100000 rows.
"""

import pandas as pd
import os
import csv
from scipy.io import loadmat

# Get all the excel files in the current directory
excel_files = [f for f in os.listdir('Data/philcar') if f.endswith('.xlsx')]

filename = 'Data/phil_socdata_train.csv'

# Open the csv file and write the column names
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['SOC', 'V', 'I', 'T','P','V_avg_five','V_avg_one','I_avg'])

    # Loop through all the excel files
    for excel_file in excel_files:

        # skip the excel file with name Phil_DC_93_10degC.xlsx as it will be used as test data
        if excel_file == 'Phil_DC_93_10degC.xlsx' or excel_file == 'Phil_DC_86_10degC.xlsx':
            continue

        # Read the excel file into a pandas dataframe
        df = pd.read_excel("Data/philcar/"+excel_file, sheet_name=None)
        
        time = []
        soc = []
        v = []
        i = []
        t = []
        v_avg_five = []
        v_avg_one = []
        i_avg = []
         = []

        # gather the values from the dataframe if they are not NaN
        for sheet in df:
            if 'HV Battery SOC [%]' in df[sheet].columns:
                soc = df[sheet]['HV Battery SOC [%]']
            if 'HV Battery Voltage [V]' in df[sheet].columns:
                v = df[sheet]['HV Battery Voltage [V]']
            if 'HV Battery Current [A]' in df[sheet].columns:
                i = df[sheet]['HV Battery Current [A]']
            if 'OAT [degC]' in df[sheet].columns:
                t = df[sheet]['OAT [degC]']

        if soc.empty or v.empty or i.empty or t.empty:
            print('Error: NaN value in file ' + excel_file)
            continue
        else:
            # print the current iteration and the file name
            print(str(excel_files.index(excel_file)) + ': Found all columns in file:' + excel_file)

            # iterate through the lists and write the data to the csv file, including the timestep of 1s using pandas 
            for x in range(len(soc)):
                # double check that the data is not NaN
                if soc[x] == soc[x] and v[x] == v[x] and i[x] == i[x] and t[x] == t[x]:
                    # v_avg_five is the average voltage of the 500 previous values including the current value
                    v_avg_five.append(sum(v[max(0, x-499):x+1])/min(500, x+1))

                    # v_avg_one is the average voltage of the 100 previous values including the current value
                    v_avg_one.append(sum(v[max(0, x-99):x+1])/min(100, x+1))

                    # i_avg is the average current of the 500 previous values including the current value
                    i_avg.append(sum(i[max(0, x-499):x+1])/min(500, x+1))

                    #writes the data to the csv file
                    writer.writerow([soc[x], v[x], i[x], t[x], v[x]*i[x], v_avg_five[x], v_avg_one[x], i_avg[x]])


# remove any rows where v < 300
df = pd.read_csv(filename)
df = df[df['V'] > 300]
df.to_csv(filename, index=False)

# remove any rows where V < 200
df = pd.read_csv(filename)
df = df[df.V > 300]
df.to_csv(filename, index=False)

print('Done!')
