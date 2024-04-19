"""
This script is used to retrieve the full stress data, creating a 2D and 3D data file to be used for training, validation, and testing of ML models.
It is assumed that the user will retrieve the data from GitHub.
Before running the code, specify the desired file save locations in lines 16 and 17.
Also ensure that 'github' is specified in line 210 (unless files are being retrieved locally, in which case use 'local')

"""

import pandas as pd
import numpy as np
import requests

# File Directories
LOCAL_FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/split_files/github/" # (if files are already on the local system)
# Adjust the following two directories to the locations where you would like the files to be stored
PROCESSED_FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/split_files/" # Where the 100 individual files will be saved
DATASET_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/" # Where the processed 2D and 3D datasets will be saved

def calculate_von_mises(S11, S22, S33, S12, S13, S23):
    """
    Calculate the von Mises stress from the specified stress components.

    """
    return np.sqrt(((S11 - S22)**2 + (S22 - S33)**2 + (S33 - S11)**2 +
                    6 * (S12**2 + S23**2 + S13**2)) / 2)

def extract_info_values(info):
    """
    Extract the last three values (travel length, travel speed, heat input) from a formatted 'info' string in each row of a DataFrame. 
    Returns None for each value if the 'info' string contains fewer than three parts.

    """
    try:
        parts = info.split('-')
        if len(parts) >= 3:
            return parts[-3], parts[-2], parts[-1]
        else:
            return None, None, None
    except Exception:
        # Return default values or handle as needed if an error occurs
        return None, None, None

def get_local_files():
    """
    Process files stored locally by: 
    - reading them, 
    - extracting key parameters from the 'info' field into new columns, 
    - calculating von Mises stress and storing in a new column,
    - removing newly obsolete columns, 
    - and saving the results to new files containing processed data. 
    Also handles cleaning of DataFrame headers and ensures numeric conversion where necessary.

    """
    file_indices = range(1, 101) # To iterate over each of the 100 files
    for i in file_indices:
        file_name = f"{LOCAL_FILE_DIRECTORY}stress_field_{i+1}.csv" # Names of the local files
        new_file = f"{PROCESSED_FILE_DIRECTORY}vonmises_stress_field_{i}.csv" # Names of the files being created
        
        df = pd.read_csv(file_name, low_memory=False) # Read the files

        df.columns = df.columns.str.strip() # Remove unwanted spaces in the column headers
        
        extracted = df['info'].apply(lambda x: extract_info_values(x)) # Extract the values from the 'info' column
        
        travel_length, travel_speed, heat_input = zip(*extracted) # Create separate series for each new value
        
        # Insert the new columns after the first column.
        df.insert(1, 'travel_length', travel_length)
        df.insert(2, 'travel_speed', travel_speed)
        df.insert(3, 'heat_input', heat_input)
        
        # Convert the new columns to numeric, handling or removing non-numeric rows as necessary
        df['travel_length'] = pd.to_numeric(df['travel_length'], errors='coerce')
        df['travel_speed'] = pd.to_numeric(df['travel_speed'], errors='coerce')
        df['heat_input'] = pd.to_numeric(df['heat_input'], errors='coerce')

        # Calculate and store von Mises stress values in a new column
        df['von_mises'] = calculate_von_mises(
            df['S11'], df['S22'], df['S33'],
            df['S12'], df['S13'], df['S23']
        )
        
        # Drop the 'info' column and the six stress columns
        columns_to_drop = ['info','S11', 'S22', 'S33', 'S12', 'S13', 'S23']
        df = df.drop(columns=columns_to_drop)

        df.to_csv(new_file, index=False) # Save the amended files
        print(f"Processed file{i}") # Print progress checks in the terminal


def get_github_files():
    """
    Process files stored at GitHub.com by: 
    - reading them, 
    - extracting key parameters from the 'info' field into new columns, 
    - calculating von Mises stress and storing in a new column,
    - removing newly obsolete columns, 
    - and saving the results to new files containing processed data. 
    Also handles cleaning of DataFrame headers and ensures numeric conversion where necessary.

    """
    base_url = "https://raw.githubusercontent.com/ed-bat/stress_predictor/main/" # Location of the GitHub files
    file_indices = range(1, 101) # To iterate over each of the 100 files
    for i in file_indices: 
        file_name = f"stress_field_{i}.csv" # Names of the files being retrieved
        url = f"{base_url}{file_name}" # Full url location for each individual file
        new_file = f"{PROCESSED_FILE_DIRECTORY}vonmises_stress_field_{i}.csv" # Names of the files being created
        
        # Attempt to download the file
        try:
            response = requests.get(url)
            response.raise_for_status()  # Will raise an HTTPError for bad requests (400+)
            
            # Read the content of the file into a pandas DataFrame
            from io import StringIO
            data = StringIO(response.text)
            df = pd.read_csv(data, low_memory=False)

            df.columns = df.columns.str.strip() # Remove unwanted spaces in the column headers
        
            extracted = df['info'].apply(lambda x: extract_info_values(x)) # Extract the values from the 'info' column
        
            travel_length, travel_speed, heat_input = zip(*extracted) # Create separate series for each new value
        
            # Insert the new columns after the first column
            df.insert(1, 'travel_length', travel_length)
            df.insert(2, 'travel_speed', travel_speed)
            df.insert(3, 'heat_input', heat_input)
            
            # Convert the new columns to numeric, handling or removing non-numeric rows as necessary
            df['travel_length'] = pd.to_numeric(df['travel_length'], errors='coerce')
            df['travel_speed'] = pd.to_numeric(df['travel_speed'], errors='coerce')
            df['heat_input'] = pd.to_numeric(df['heat_input'], errors='coerce')

            # Calculate and store von Mises stress values in a new column
            df['von_mises'] = calculate_von_mises(
                df['S11'], df['S22'], df['S33'],
                df['S12'], df['S13'], df['S23']
            )
        
            # Drop the 'info' column and the six stress columns
            columns_to_drop = ['info','S11', 'S22', 'S33', 'S12', 'S13', 'S23']
            df = df.drop(columns=columns_to_drop)

            df.to_csv(new_file, index=False) # Save the amended files
            print(f"Processed file{i}") # Print progress checks in the terminal

        # Error handling
        except requests.exceptions.HTTPError as err:
            print(f"Failed to retrieve {file_name}: {err}")
        except Exception as e:
            print(f"An error occurred: {e}")


def create_dataset(dimension):
    """
    Consolidate processed stress field files into a single dataset file, either in 2D or 3D format, based on the specified dimension. 
    Filters out rows based on the 'X' column criteria specific to 2D or 3D requirements and saves the concatenated result into a CSV file. 
    Ensures the removal of specific columns in the 2D dataset.

    """
    file_indices = range(1, 101) # To iterate over each of the 100 files
    dfs = [] # Create empty dataframe ready to be populated with each .csv file
    for i in file_indices:
        new_file = f"{PROCESSED_FILE_DIRECTORY}vonmises_stress_field_{i}.csv" # Location of processed data files
        df = pd.read_csv(new_file) # Read processed data files
        
        if dimension == '2D':
            df = df[df["X"] == -0.09] # Specify x-coordinate for the 2D cross-section
        elif dimension == '3D':
            df = df[df["X"] != ' X']  # Filter out erroneous header rows hidden within the file

        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True) # Concatenate each individual file into a single dataframe

    # Remove the 'X' column for the 2D data file
    if dimension == '2D':
        combined_df.drop(columns=['X'], inplace=True, errors='ignore')

    combined_df.to_csv(f"{DATASET_DIRECTORY}{dimension}_vonmises_stress_field.csv", index=False) # Save the data file
    print(f'{dimension} dataset created.') # Print progress check in the terminal



def data_preparation(location_type):
    """
    Orchestrates the data preparation process by: 
    selecting the source of the files (local or GitHub), 
    processing the files to compute von Mises stress and other parameters, 
    and finally generating consolidated 2D and 3D datasets. 
    Outputs status messages to track the progress of file processing and dataset creation.

    """
    # Check the file location and retrieve files accordingly
    if location_type == 'local':
        get_local_files()
    elif location_type == 'github':
        get_github_files()

    print('Individual files created. Creating combined data file...') 

    create_dataset('2D')
    print('2D file created. Creating 3D data file...')
    create_dataset('3D')
    print('Done')

#---------------------------------------------------------------------------------------------------------------------------------------------------------

location_type = 'github' # Replace 'github' with 'local' if using files on your local system and ensure correct directory to your files

data_preparation(location_type) # Run the program