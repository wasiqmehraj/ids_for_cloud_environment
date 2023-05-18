
import csv
from datetime import datetime, timedelta

# parse the timestamp into the date time object
def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, '%H:%M:%S')

# format the timestamp to add the date
def format_timestamp(timestamp):
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

# Open the CSV file and read the data
with open('2_combined_slave_data.csv', newline='') as csvfile:
    # print('succesfully opened')
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the first row (header)
    data = list(reader)

# Loop through the data and modify the timestamps
for i in range(len(data)):
    print('in loop')
    timestamp_str = data[i][1]   # Get the timestamp string
    date_str = data[i][8][:10]  # Get the date from the VM name

    timestamp = parse_timestamp(timestamp_str)   # Parse the timestamp string to a datetime object
    new_timestamp = timestamp.replace(year=int(date_str[:4]), month=int(date_str[5:7]), day=int(date_str[8:])) + timedelta(hours=8) # Add 8 hours to the timestamp and replace the date
    # print(f'The new timestamp is {new_timestamp}')
    data[i][1] = format_timestamp(new_timestamp)     # Format the new timestamp as a string and replace the old one

    percent_done =  i/len(data)
    # print(f'{percent_done} % done.')

# sort by timestamp
data.sort(key=lambda x: datetime.strptime(x[1], '%Y-%m-%d %H:%M:%S'))

# Write the modified data to a new CSV file
with open('3_time_adjusted.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # Write the header row
    writer.writerows(data)
