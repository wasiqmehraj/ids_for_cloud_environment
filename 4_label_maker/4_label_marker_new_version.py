import csv
from datetime import datetime, time

# Start and end times that are malicious according to the documentation
malicious_times = [
    (datetime.strptime('2016-12-16 17:09:45', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:12:13', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-16 17:16:15', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:16:42', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-16 17:19:44', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:23:59', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-16 17:26:31', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:53:25', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-16 17:32:22', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:33:06', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-16 17:36:26', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:36:26', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-16 17:47:07', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-16 17:52:26', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-19 17:33:27', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-19 18:01:24', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-19 17:34:36', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-19 17:34:53', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-19 17:36:18', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-19 17:36:18', '%Y-%m-%d %H:%M:%S')),
    (datetime.strptime('2016-12-19 17:57:00', '%Y-%m-%d %H:%M:%S'), datetime.strptime('2016-12-19 18:00:00', '%Y-%m-%d %H:%M:%S')),

]


# Function to check if the function is malicious ie. if the timestamp coincides within the malicious times or not
def is_malicious(timestamp):
    for start, end in malicious_times:
        if start <= timestamp <= end:
            return True
    return False


# Add the label accordingly
with open('3_time_adjusted.csv', 'r') as input_file, open('4_made_label.csv', 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    # Write the header row to the output file
    header_row = next(reader)
    header_row.insert(9, 'label')
    # Remove the columns from the header row
    header_row.pop(4)  # remove 'syscall-arg2'
    # header_row.pop(1)  # remove 'time-stamp'
    header_row.pop(0)  # remove 'PID'
    writer.writerow(header_row)
    for row in reader:
        # Check if the timestamp is malicious and add the appropriate value to the 9th column
        timestamp = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        if is_malicious(timestamp):
            row.insert(9, 'malicious')
        else:
            row.insert(9, 'benign')
        # Remove the columns from the row
        row.pop(4)  # remove 'syscall-arg2'
        # row.pop(1)  # remove 'time-stamp'
        row.pop(0)  # remove 'PID'
        # Write the updated row to the output file
        writer.writerow(row)
