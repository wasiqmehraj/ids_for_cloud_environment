import pandas as pd
from sklearn.preprocessing import StandardScaler

input_data = r'D:\ids_for_cloud_env\7_condense_batch\condensed_batches_1sec.csv'
# Load the dataset
data = pd.read_csv(input_data)

# Select the columns to be standardized
columns_to_standardize = data.columns[1:-1]

# Standardize the data using StandardScaler
scaler = StandardScaler()
data[columns_to_standardize] = scaler.fit_transform(data[columns_to_standardize])

# DATA Cleaning
# Check for missing values
print('Number of missing values:', data.isna().sum().sum())

# Replace missing values with the mean of the column
data.fillna(data.mean(), inplace=True)

# Save the standardized data to a new CSV file
data.to_csv('standardized_clean_output_1sec.csv', index=False,float_format='%.10f')
