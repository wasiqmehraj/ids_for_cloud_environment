import os
import csv
import datetime

# Define the time interval for each batch (in seconds)
batch_interval = 1

# Set the input and output directory paths
input_dir = r'D:\project NIT\consolidated_try\6_split_wrt_vms'
output_dir = r'D:\project NIT\consolidated_try\7_split_wrt_vms_batches'

# Iterate through all the input files in the directory
for input_file in os.listdir(input_dir):
    # Skip any files that are not CSV files
    if not input_file.endswith('.csv'):
        continue

    # Construct the full paths for the input and output files
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, f'{input_file[:-4]}_batches.csv')

    # Open the input CSV file and read the data
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]

    # Initialize variables for batch processing
    current_batch_start = None
    current_batch_num = 0

    # Loop through the data and split it into batches
    for example in data:
        # Parse the timestamp string into a datetime object
        timestamp = datetime.datetime.strptime(example[0], '%Y-%m-%d %H:%M:%S')

        # If we haven't started a batch yet, or if the current example
        # falls outside the current batch's time range, start a new batch
        if current_batch_start is None or timestamp >= current_batch_start + datetime.timedelta(seconds=batch_interval):
            current_batch_start = timestamp
            current_batch_num += 1

        # Insert the batch number as a new column at index 9 (i.e., the 10th column)
        example.insert(9, current_batch_num)

    # Open the output CSV file and write the data
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header.insert(9, 'batch-number')
        writer.writerow(header)
        writer.writerows(data)
