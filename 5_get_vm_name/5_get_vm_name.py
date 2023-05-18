import csv
import re

filename = '4_made_label.csv'  
output_filename = '5_vm_name_obtained.csv' 

# compile regex pattern
pattern = re.compile(r'.*uvic\.ca_(.+?)\.syscalls')


with open(filename, 'r') as csv_file, open(output_filename, 'w', newline='') as output_file:
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(output_file)

    for i, row in enumerate(csv_reader):
        # print('inside loop')

        if i == 0:
            # print('inside if')
            # write header row to output file
            csv_writer.writerow(row)
        else:
            # extract actual name from 6th column
            filename = row[6]
            # print(f'filename{filename}')
            match = pattern.match(filename)
            if match:
                actual_name = match.group(1)
            else:
                # print(f'not a match{i}')
                actual_name = ''

            # remove extension
            actual_name = actual_name.split('.')[0]

            # update row with actual name
            row[6] = actual_name

            # write updated row to output file
            csv_writer.writerow(row)
