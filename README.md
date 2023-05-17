# **Intrusion Detection System for Cloud Environment**



This project, titled "Intrusion Detection System for Cloud Environment," serves as my master's dissertation at the National Institute of Technology, Srinagar. It aims to detect and prevent unauthorized access or malicious activities within the cloud system. The system utilizes various components and tools to achieve its objectives.

## __System Calls__
System calls are functions provided by the operating system that allow programs to interact with the system's resources. They provide an interface for processes to request services from the operating system. In this project, we make use of system calls to monitor and analyze the behavior of processes within the cloud environment.

## __Components__

### __lz4 Decompressor__
The system includes an lz4 decompressor that converts lz4-compressed files to .syscall trace files. This conversion process allows for easier analysis and extraction of system calls.


### __Extractor__
The extractor component is responsible for extracting the required system calls from the .syscall trace file. It analyzes the trace file and extracts the relevant information, storing it in a CSV (Comma-Separated Values) format. This extracted data serves as the basis for further analysis and processing.


### __Concatenator__
The system provides a concatenator.py script that concatenates multiple CSV files. This script is particularly useful when running multiple instances of the extractor component in parallel. By combining the resulting CSV files, a comprehensive dataset can be created for analysis and model training.


### __Time Adjuster__
The time_adjust.py script adjusts the timezone of the timestamps within the system call data. This ensures consistency and facilitates chronological sorting of the events. By organizing the data in chronological order, it becomes easier to identify patterns and anomalies.


### __Label Maker__
The label_maker.py script assigns labels to the system call data based on the provided documentation. It categorizes the events as either benign or malignant, aiding in the classification and identification of potential security threats.


### __VM Name Extractor__
The get_vm_name.py script extracts the virtual machine (VM) name from the filename of the syscall trace file. This information is valuable for associating system call data with specific VM instances within the cloud environment.


### __Batch Creator__
The make_batches.py script marks batch numbers within a specific time period. This process partitions the data into batches, which can be useful for analyzing the behavior of processes over time and identifying patterns within specific time intervals.


### __Batch Condenser__
The condense_batches.py script condenses individual batches into a single entry. This condensation facilitates feeding the data to the model as one example, simplifying the input representation and potentially improving the model's performance.


## __Preprocessing Scripts__
The project includes several Python files that handle preprocessing tasks before feeding the data to the actual model. These preprocessing steps may include data cleaning, feature extraction, or normalization, depending on the requirements of the chosen model.

## __Model Implementation__
The code includes the following model implementation files:

__logistic_regression.py__: Implements a logistic regression model for intrusion detection. The model uses 5-fold cross-validation and calculates metrics such as accuracy, recall, precision, and AUC score. It also generates an ROC plot for evaluation purposes.

__random_forest.py__: Implements a random forest model for intrusion detection. Similar to the logistic regression model, it uses 5-fold cross-validation and computes various metrics to assess its performance. Additionally, it generates an ROC plot.

__neural_network.py__: Implements a neural network model for intrusion detection. This model also employs 5-fold cross-validation and calculates accuracy, recall, precision, AUC score, and generates an ROC plot.


## __Data Source__
The dataset used for this project is the ISOT-CID dataset from the University of Victoria. It provides a collection of system call traces that can be used for intrusion detection and analysis purposes.

Please note that, for simplicity, we have only utilized read and write syscalls in the extractor component. However, it is possible to add other system calls by indicating their corresponding patterns in the extractor's configuration.

## __Usage__
To use this intrusion detection system, follow these steps:

Clone or download the project repository.
Ensure that the required dependencies are installed (specified in the project's documentation).
Run the necessary scripts in the provided order to perform decompression, extraction, concatenation, time adjustment, label assignment, VM name extraction, batch creation, batch condensation, preprocessing, and model implementation.
Review the generated metrics, ROC plots, and any other relevant output to assess the performance of the implemented models.
For detailed instructions and examples, please refer to the project's documentation.

## __Credits__
This project was developed by **Wasiq Mehraj** (wasiqmehraj_ece21@nitsri.net) and is based on the ISOT-CID dataset from the University of Victoria. Special thanks to the contributors and maintainers of the dataset for their valuable resources. This project was made under the guidance of **Dr. Mohammad Ahsan Chesti** and **Dr. G.R. Begh**.

