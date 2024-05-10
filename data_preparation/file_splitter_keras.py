"""
file_splitter_keras.py

Script used to process the full datasets into 20 files to be utilised by Keras models

"""

#Specify the location of the stress file to process
FILE_PATH = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/3D_stress_field.csv" 

#Specify the location to save the split files
OUTPUT_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/split_files_3D/"

#Specify which columns are not required
COLUMNS_TO_EXCLUDE = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']

#---------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import os
from math import ceil
from sklearn.utils import shuffle


def shuffle_and_split(file_path, base_output_directory, excluded_columns=None, number_of_files=20):
    
    df = pd.read_csv(file_path)

    # Remove unnecessary columns
    if excluded_columns:
        df.drop(columns=excluded_columns, inplace=True)

    # Shuffle the DataFrame
    df = shuffle(df, random_state=42)

    # Create base output directory if it doesn't exist
    if not os.path.exists(base_output_directory):
        os.makedirs(base_output_directory)
    
    # Define directories for training, validation, and test sets
    train_dir = os.path.join(base_output_directory, 'train')
    validation_dir = os.path.join(base_output_directory, 'validation')
    test_dir = os.path.join(base_output_directory, 'test')

    # Create these directories if they do not exist
    for directory in [train_dir, validation_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    total_rows = len(df)
    rows_per_file = ceil(total_rows / number_of_files)  # Calculate the number of rows per file

    # Define the number of files for training, validation, and test sets
    num_train = 13
    num_validation = 3
    num_test = 4

    # Prepare DataFrame segments for training stats
    train_dfs = []
    test_dfs = []

    # Split and save the files
    file_index = 0
    for i in range(num_train):
        train_df = save_split(df, file_index, rows_per_file, total_rows, train_dir, 'train', i)
        train_dfs.append(train_df)
        file_index += 1

    for i in range(num_validation):
        save_split(df, file_index, rows_per_file, total_rows, validation_dir, 'validation', i)
        file_index += 1

    for i in range(num_test):
        test_df = save_split(df, file_index, rows_per_file, total_rows, test_dir, 'test', i)
        test_dfs.append(test_df)
        file_index += 1

    # Aggregate training data and calculate stats
    train_full_df = pd.concat(train_dfs)
    train_stats = train_full_df.describe().loc[['mean', 'std']]
    test_full_df = pd.concat(test_dfs)
    test_stats = test_full_df.describe().loc[['mean', 'std']]

    # Save the stats to a file
    train_stats_file_name = os.path.join(OUTPUT_DIRECTORY, 'training_stats.csv')
    train_stats.to_csv(train_stats_file_name)
    print(f'Saved overall training set statistics to {train_stats_file_name}')
    test_stats_file_name = os.path.join(OUTPUT_DIRECTORY, 'test_stats.csv')
    test_stats.to_csv(test_stats_file_name)
    print(f'Saved overall test set statistics to {test_stats_file_name}')

    print(f'Done. Split the file into {file_index} smaller files across training, validation, and test sets.')

def save_split(df, file_index, rows_per_file, total_rows, directory, set_name, i):
    start = file_index * rows_per_file  # Calculate starting row index for each split
    end = min((file_index + 1) * rows_per_file, total_rows)  # Calculate ending row index for each split
    
    # Create a new filename for each split
    new_file_name = os.path.join(directory, f'{set_name}_{i+1}.csv')
    split_df = df.iloc[start:end]
    
    # Save the split to a new file
    split_df.to_csv(new_file_name, index=False)
    print(f'Saved {new_file_name}')  # Track progress in terminal

    return split_df  # Return the DataFrame for aggregation

# --------------------------------------------------------------------------------------------------------------------

shuffle_and_split(FILE_PATH, OUTPUT_DIRECTORY, excluded_columns=COLUMNS_TO_EXCLUDE)