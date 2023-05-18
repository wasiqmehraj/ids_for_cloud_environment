import re
import pandas as pd
from tqdm import tqdm
import os

folder_path = r"D:\project NIT\Phase I\syscalls\2016-12-16\one\slave12"
file_extension = ".syscalls"

# get a list of all files in the folder
file_list_all = os.listdir(folder_path)

# filter the list to only include files with the specified extension
file_list = [file_name for file_name in file_list_all if file_name.endswith(file_extension)]

# Define the regular expression pattern to match syscall lines
pattern = r'^(?P<PID>\d+)\s+(?P<Timestamp>\d+:\d+:\d+)\s+(?P<Syscall_name>\w+)\((?P<Argument1>[^)]+), (?P<Argument2>[^)]+), (?P<Argument3>[^)]+)\)\s+=\s+(?P<Return_value>-?\d+)\s+<(?P<Time_elapsed>[\d\.]+)>$'

# Define the system call names to match
syscall_names = ['read', 'write']

# Open the syscall trace file for reading
count = 0
for file_name in file_list:
    no_of_files = len(file_list)
    count += 1
    print(f'Reading file {count} of {no_of_files}, {(count*100)/no_of_files}% completed')
    with open((f"{folder_path}\{file_name}"), 'r') as f:
        # Initialize an empty list to store the extracted data
        data = []

        # Iterate over the lines in the file
        for line in tqdm(f, desc=f"Processing {file_name}", unit=" lines"):
            # Match the pattern against the line
            match = re.match(pattern, line)

            if match:
                # Extract the syscall name
                syscall_name = match.group(3)

                # Check if the syscall name is in the list of names to match
                if syscall_name in syscall_names:

                    # Extract the other matched fields
                    pid = int(match.group(1))
                    timestamp = match.group(2)
                    syscall_arg1 = int(match.group(4))
                    syscall_arg2 = match.group(5)
                    syscall_arg3 = int(match.group(6))
                    return_value = int(match.group(7))
                    time_elapsed = float(match.group(8))

                    # Create a dictionary with the extracted fields and add the filename
                    data_dict = {
                        'PID': pid,
                        'time-stamp': timestamp,
                        'syscall-name': syscall_name,
                        'syscall-arg1': syscall_arg1,
                        'syscall-arg2': syscall_arg2,
                        'syscall-arg3': syscall_arg3,
                        'return-value': return_value,
                        'time-elapsed': time_elapsed,
                        'vm-name': file_name
                    }

                    # Append the dictionary to the data list
                    data.append(data_dict)

# Create a pandas DataFrame from the data list
print('Count is ', count)
df = pd.DataFrame(data)
df.to_csv('16-12-16_slave12a.csv', index=False)
