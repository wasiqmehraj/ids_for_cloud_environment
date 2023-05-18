import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_data = r'D:\ids_for_cloud_env\7_condense_batch\condensed_batches_1sec.csv'
# Load the dataset
data = pd.read_csv(input_data)

# Extract the columns to be normalized
cols_to_normalize = data.columns[1:-1]

# Normalize the data in each column
scaler = MinMaxScaler()
data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])

# DATA Cleaning
# Check for missing values
print('Number of missing values:', data.isna().sum().sum())

# Replace missing values with the mean of the column
data.fillna(data.mean(), inplace=True)

# Save the normalized data to a new CSV file
data.to_csv('normalized_clean_output_1sec.csv', index=False, float_format='%.10f')
