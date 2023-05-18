import lz4.frame
import os

input_dir = r"D:\project NIT\syscalls\2016-12-19"
output_dir = r"D:\project NIT\new_syscalls\2016-12-19"

file_extension = ".lz4"

# get a list of all files in the folder
file_list_all = os.listdir(input_dir)
# print(file_list_all)

# filter the list to only include files with the specified extension
file_list = [file_name for file_name in file_list_all if file_name.endswith(file_extension)]
length = len(file_list)


count = 0
for file_name in file_list:
    count += 1
    percentage = (count/length)*100
    print(f'{percentage} Completed.\n COUNT: {count} of {length}\n ')

    # Decompress the data
    with lz4.frame.open((f"{input_dir}\{file_name}"), mode='r') as fp:
        output_data = fp.read()
        # print('The output data is ',output_data)
        data = []

        # conversion to a string 'utf-8'
        output_str = output_data.decode('utf-8')

        # adding the extension
        output_path = os.path.join(output_dir, f"{file_name}.syscalls")
        # print('The final output path is ', output_path)

        # Write the string to a file
        with open(output_path, "w") as file:
            file.write(output_str)
