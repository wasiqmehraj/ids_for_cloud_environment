import pandas as pd

in_file_path = r"D:\ids_for_cloud_env\6_make_batches\batches_full_file_1sec.csv"
# read the CSV file into a pandas dataframe
df = pd.read_csv(in_file_path)

# remove the time-stamp column
df.drop('time-stamp', axis=1, inplace=True)

# initialize empty dictionary to store results for each batch
results = {}

# iterate through each batch number
for batch_num in df['batch-number'].unique():
    print(f'\nCurrently processing batch number {batch_num}. \n')
    # filter dataframe for the current batch number
    batch_df = df[df['batch-number'] == batch_num]

    # count read and write entries
    read_count = batch_df[batch_df['syscall-name'] == 'read'].shape[0]
    write_count = batch_df[batch_df['syscall-name'] == 'write'].shape[0]

    # count fd values
    fd_4 = batch_df[batch_df['syscall-arg1'] == 4].shape[0]
    fd_5 = batch_df[batch_df['syscall-arg1'] == 5].shape[0]
    fd_6 = batch_df[batch_df['syscall-arg1'] == 6].shape[0]
    fd_29 = batch_df[batch_df['syscall-arg1'] == 29].shape[0]
    fd_n = batch_df[~batch_df['syscall-arg1'].isin([4, 5, 6, 29])].shape[0]

    # compute read and write max values
    read_max = batch_df[batch_df['syscall-name'] == 'read']['syscall-arg3'].max()
    write_max = batch_df[batch_df['syscall-name'] == 'write']['syscall-arg3'].max()

    # compute read and write return values
    read_got = batch_df[batch_df['syscall-name'] == 'read']['return-value'].sum()
    write_got = batch_df[batch_df['syscall-name'] == 'write']['return-value'].sum()

    # compute read and write time elapsed
    read_time = batch_df[batch_df['syscall-name'] == 'read']['time-elapsed'].sum()
    write_time = batch_df[batch_df['syscall-name'] == 'write']['time-elapsed'].sum()

    # count total rows and malicious rows
    total_entries_batch = batch_df.shape[0]
    malicious_count_batch = batch_df[batch_df['label'] == 'malicious'].shape[0]

    # calculate percent malicious
    percent_malicious = malicious_count_batch / total_entries_batch

    # determine if the batch is malicious or not
    # 1 ----> Malicious
    # 0 ----> Benign
    if percent_malicious >= 0.25:
        is_malicious = '1'
    else:
        is_malicious = '0'

    # store results in dictionary for current batch
    results[batch_num] = {'read-count': read_count,
                          'write-count': write_count,
                          'fd-4': fd_4,
                          'fd-5': fd_5,
                          'fd-6': fd_6,
                          'fd-29': fd_29,
                          'fd-n': fd_n,
                          'read-max': read_max,
                          'write-max': write_max,
                          'read-got': read_got,
                          'write-got': write_got,
                          'read-time': read_time,
                          'write-time': write_time,
                          'total-entries-batch': total_entries_batch,
                          'is-malicious': is_malicious
                          }
# create a new pandas dataframe to store the results
result_df = pd.DataFrame(
    columns=['batch-number', 'read-count', 'write-count', 'fd-4', 'fd-5', 'fd-6', 'fd-29', 'fd-n', 'read-max',
             'write-max', 'read-got', 'write-got', 'read-time', 'write-time', 'total-entries-batch', 'is-malicious'])

# iterate through each batch and add the results to the dataframe
for batch_num, values in results.items():
    result_df = result_df._append({'batch-number': batch_num,
                                  'read-count': values['read-count'],
                                  'write-count': values['write-count'],
                                  'fd-4': values['fd-4'],
                                  'fd-5': values['fd-5'],
                                  'fd-6': values['fd-6'],
                                  'fd-29': values['fd-29'],
                                  'fd-n': values['fd-n'],
                                  'read-max': values['read-max'],
                                  'write-max': values['write-max'],
                                  'read-got': values['read-got'],
                                  'write-got': values['write-got'],
                                  'read-time': values['read-time'],
                                  'write-time': values['write-time'],
                                  'total-entries-batch': values['total-entries-batch'],
                                  'is-malicious': values['is-malicious']}, ignore_index=True)

    # write the dataframe to a CSV file
    result_df.to_csv('condensed_batches_1sec_25%.csv', index=False)
