"""
This program will go through every excel file in the folder Data/philcar it will parse the data as specified and write it to a csv file

Parsing Specifications:
The program will gather the following information from the second sheet in each excel file:
    - Initial SOC
    - Final SOC
    - Initial Altitude
    - Final Altitude
    - Average Speed
    - Final Accumulated Distance

To gather this data the program will break up each excel file into chunks between 10 and 100 minutes long (600 - 6000 rows). 
These chunks can overlap with each other by half the chunk size.

The program will then calculate the following data before writing it to a csv file:
    - Initial SOC (First value of SOC in the chunk)
    - Final SOC (Last value of SOC in the chunk)
    - Altitude Difference (Final Altitude - Initial Altitude)
    - Average Speed (Average of all the speeds in the chunk)
    - Final Accumulated Distance (Final Distance - Initial Distance)

"""

import os
import csv
import pandas as pd
import numpy as np


def get_chunks(data, chunk_size, overlap) -> list:
    """This function will break up the data into chunks of chunk_size long, with overlap between the chunks.

    Args:
        data (list): The data to be broken up into chunks.
        chunk_size (int): The size of each chunk.
        overlap (int): The amount of overlap between each chunk.

    Returns:
        list: A list of lists, where each list is a chunk of data.
    """
    chunks = []
    for x in range(0, len(data), chunk_size - overlap):
        if x + chunk_size < len(data):
            chunks.append(data[x:x + chunk_size])
    return chunks   



def process_chunk(chunk) -> list:
    """This function will process a chunk of data.

    Args:
        chunk (list): A list of data.

    Returns:
        list: A list of the processed data.
    """
    # Get the data from the chunk
    soc = chunk['HV Battery SOC [%]']
    altitude = chunk['Altitude_filt [m]']
    speed = chunk['Vehicle Speed [m/s]']
    distance = chunk['Accumulated_Distance [m]']

    # Calculate the data
    initial_soc = soc.iloc[0]
    final_soc = soc.iloc[-1]
    altitude_difference = altitude.iloc[-1] - altitude.iloc[0]
    average_speed = np.mean(speed)
    final_accumulated_distance = distance.iloc[-1] - distance.iloc[0]

    # Return the data
    return [initial_soc, final_soc, altitude_difference, average_speed, final_accumulated_distance]

def main():

    # Get all the excel files in the current directory
    excel_files = [f for f in os.listdir('Data/philcar') if f.endswith('.xlsx')]

    # Create a new csv file to write to
    with open('Data/phil_rangedata_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['initial_SOC', 'final_SOC', 'altitude', 'avg_speed', 'total_distance'])

        # Loop through all the excel files
        for excel_file in excel_files:

            # skip the excel file with name Phil_DC_93_10degC.xlsx as it will be used as test data
            if excel_file == 'Phil_DC_93_10degC.xlsx' or excel_file == 'Phil_DC_86_10degC.xlsx':
                continue

            # Read the excel file into a pandas dataframe
            df = pd.read_excel('Data/philcar/' + excel_file, sheet_name=1)

            previous_row = []

            # Get the chunks of data for 10 mins, 30 mins, 60 mins, 90 mins, 120 mins, and 150 mins and the entire chunk
            for chunk_size in [600, 1800, 3600, 5400, 7200, 9000, len(df)-1]:
                overlap = int(chunk_size / 2)

                if chunk_size > len(df):
                    continue

                chunks = get_chunks(df, chunk_size, overlap)

                # print the current status
                print(f'Processing {excel_file} with chunk size {chunk_size} and overlap {overlap}')

                # Loop through all the chunks
                for i in range(len(chunks)):
                    # Process the chunk
                    processed_chunk = process_chunk(chunks[i])

                    # check if all the values in the chunk are not NaN and the final accumulated distance is not the same as the row above it in the csv file
                    if not np.isnan(processed_chunk).any() and processed_chunk != previous_row:
                        # Write the only the first row of the processed chunk to the csv file
                        writer.writerow(processed_chunk)
                        previous_row = processed_chunk

            # Gather the data for the entire file
            processed_chunk = process_chunk(df)

            # check if all the values in the chunk are not NaN and the final accumulated distance is not the same as the row above it in the csv file
            if not np.isnan(processed_chunk).any() and processed_chunk != previous_row:
                # Write the processed chunk to the csv file
                writer.writerow(processed_chunk)
                previous_row = processed_chunk

                


                            
if __name__ == '__main__':
    main()

    '''
    #randomly reformat the rows in the csv file
    df = pd.read_csv('Data/phil_rangedata_train.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('Data/phil_rangedata_train.csv', index=False)
    '''

    print('Done')
    