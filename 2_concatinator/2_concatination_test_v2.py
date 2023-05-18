import os
import pandas as pd

# input file address
directory = r"D:\project NIT\consolidated_try\individual_slaves_folder"

# getting the csv's filenames and storeing them in variable csv_files
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# sort
csv_files.sort()

# initialize the dataframe to store data
concatenated_data = pd.DataFrame()

# loop through all CSV files to concatenate
for i, file in enumerate(csv_files):
    data = pd.read_csv(os.path.join(directory, file))
    # print('here ', data)

    # skip the header (first row) for all but the first CSV file
    if i > 0:
        data = data.iloc[1:]

    # concatenate
    concatenated_data = pd.concat([concatenated_data, data])
    # print('\nconcatenated data : ', concatenated_data)

# export to csv file
concatenated_data.to_csv("2_combined_slave_data.csv", index=False)
