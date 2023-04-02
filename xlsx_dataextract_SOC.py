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

# Get all the excel files in the current directory
excel_files = [f for f in os.listdir('Data/philcar') if f.endswith('.xlsx')]

with open('Data/phil_socdata_trainLSTM.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['item_id','timestamp','SOC', 'V', 'I', 'T'])

    # Loop through all the excel files
    for excel_file in excel_files:

        # skip the excel file with name Phil_DC_93_10degC.xlsx as it will be used as test data
        if excel_file == 'Phil_DC_93_10degC.xlsx':
            continue

        # Read the excel file into a pandas dataframe
        df = pd.read_excel("Data/philcar/"+excel_file, sheet_name=None)

        time = []
        soc = []
        v = []
        i = []
        t = []

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
            print('Found all columns in file ' + excel_file)

            # iterate through the lists and write the data to the csv file, including the timestep of 1s using pandas 
            for x in range(len(soc)):
                # double check that the data is not NaN
                if soc[x] == soc[x] and v[x] == v[x] and i[x] == i[x] and t[x] == t[x]:
                    timestep = x
                    #converts timestep into a pandas timestamp
                    timestep = pd.to_datetime(timestep, unit='s')
                    #converts the timestamp into a string
                    timestep = timestep.strftime('%Y-%m-%d %H:%M:%S')
                    #writes the data to the csv file
                    writer.writerow(['SOC',timestep, soc[x], v[x], i[x], t[x]])
    
print('Done!')
