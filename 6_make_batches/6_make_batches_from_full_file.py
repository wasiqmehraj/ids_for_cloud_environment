import pandas as pd

# specify the correct data type for each column
dtypes = {'time-stamp': str, 'syscall-name': str, 'syscall-arg1': str,
          'syscall-arg3': str, 'return-value': str, 'time-elapsed': float,
          'vm-name': str, 'label': str}
input_file = r'D:\ids_for_cloud_env\5_get_vm_name\5_vm_name_obtained.csv'

# read the CSV file with the specified dtypes
df = pd.read_csv(input_file, dtype=dtypes)

# convert the time-stamp column to datetime format
df['time-stamp'] = pd.to_datetime(df['time-stamp'])
df = df.sort_values('time-stamp')

# set the batch interval to 1 second
batch_interval = pd.Timedelta(seconds=1)


df['batch-number'] = 1

# initialize the batch number and the start time
batch_number = 1
start_time = df.iloc[0]['time-stamp']

# iterate over the rows in the dataframe
for index, row in df.iterrows():
    # calculate the time elapsed since the start time
    time_elapsed = row['time-stamp'] - start_time

    # check if the time elapsed is greater than the batch interval
    if time_elapsed >= batch_interval:
        # update the batch number and start time
        batch_number += 1
        start_time = row['time-stamp']

    # assign the batch number to the current row
    df.at[index, 'batch-number'] = batch_number

# save the output to a new CSV file
df.to_csv('batches_full_file_1sec.csv', index=False)
