"""
data_preparation.py

This script is used to retrieve the full stress data, creating two data files to be used for training, validation, and testing of ML models.
The first file contains data for a 2D cross-section in the YZ plane, with the section's x-coordinate specified in line 23.
The second file contains the full stress data (please note that this file will be too large to open via Excel).

It is assumed that the user will be retrieving the data directly from GitHub (please note that the runtime will be dependent on internet speed and stability).
In the above case, ensure that 'github' is specified in line 17 (unless files are being retrieved locally, in which case use 'local' and specify their location in line 18)
Before running the code, specify the desired file save location in line 21.

Runtime may take up to 20 minutes. Progress can be tracked from the terminal.

"""

# File Directories
INPUT_FILES = 'local' # Replace 'github' with 'local' if using files on your local system and ensure correct directory for your files (line 18)
LOCAL_FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/original_data/github_files/" # (if files are already on the local system)

# Adjust the following directory to the location where you would like the files to be stored
OUTPUT_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/" # Where the processed 2D and 3D datasets will be saved

X_CROSS_SECTION_COORD = 0 # Specify desired x-coordinate location for the 2D cross-section data (YZ plane)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import requests
from io import StringIO

def calculate_von_mises(S11, S22, S33, S12, S13, S23):
    """
    Calculates the von Mises stress from the six stress components.

    """
    return np.sqrt(((S11 - S22)**2 + (S22 - S33)**2 + (S33 - S11)**2 +
                    6 * (S12**2 + S23**2 + S13**2)) / 2)

def extract_info_values(info):
    """
    Extracts the last three values (travel length, travel speed, heat input) from a formatted 'info' string in each row of the DataFrame. 
    Returns None for each value if the 'info' string contains fewer than three parts.

    """
    try:
        parts = info.split('-') # Split up the numbers from the 'info' string
        if len(parts) >= 3:
            return parts[-3], parts[-2], parts[-1] # Return the last three digits from the 'info' string
        else:
            return None, None, None
    except Exception:
        # Return default values or handle as needed if an error occurs
        return None, None, None
    
def process_dataframe(df, dimension):
    """
    Processes the DataFrame by: 
    - formatting headers, 
    - extracting input parameters, 
    - calculating von Mises stress,
    - dropping obsolete 'info' column
    - filtering the data
    - and preparing for saving. 
    Returns the processed DataFrame.
    """
    print(f'Processing {dimension} data file...')
    df.columns = df.columns.str.strip()  # Remove unwanted spaces in the column headers
    
    extracted = df['info'].apply(extract_info_values)  # Extract the values from the 'info' column
    travel_length, travel_speed, heat_input = zip(*extracted)  # Create separate series for each new value

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

    if dimension == '2D': # For the 2D data file
        df = df[df["X"] == X_CROSS_SECTION_COORD] # Store only the data at the specified cross-section (in the YZ plane)
        df.drop(columns=['X'], inplace=True, errors='ignore') # Subsequently drop the 'X' column since each entry is just 0

    elif dimension == '3D': # For the 3D data files
        df = df[df["X"] != ' X']  # Filter out erroneous header rows hidden within the file

    df.drop(columns='info', inplace=True) # Drop the 'info' column
    print(f'Successfully processed {dimension} data file.')
    return df # Return the processed dataframe


def get_local_files():
    """
    Retrieves the data files locally.
    Concatenates all 100 data files into a single dataframe.
    Returns the concatenated dataframe.
    
    """
    dfs = [] # Create empty dataframe ready to be populated with each .csv file
    file_indices = range(1, 101)  # To iterate over each of the 100 files
    for i in file_indices:
        file_name = f"{LOCAL_FILE_DIRECTORY}stress_field_{i}.csv"  # Name of the local file being retrieved
        print(f'Retrieving file {i}/100')
        df = pd.read_csv(file_name, low_memory=False)  # Read the file data
        dfs.append(df) # Add the data to the new dataframe
        print(f'file {i} retrieved successfully')

    combined_df = pd.concat(dfs, ignore_index=True) # Concatenate into a single dataframe
    return combined_df # Return the concatenated dataframe


def get_github_files():
    """
    Retrieves the data files from GitHub.
    Concatenates all 100 data files into a single dataframe.
    Returns the concatenated dataframe.
    
    """
    base_url = "https://raw.githubusercontent.com/ed-bat/stress_predictor/main/"  # Location of the GitHub files
    dfs = [] # Create empty dataframe ready to be populated with each .csv file
    file_indices = range(1, 101)  # To iterate over each of the 100 files
    for i in file_indices:
        file_name = f"stress_field_{i}.csv"  # Name of the file being retrieved
        url = f"{base_url}{file_name}"  # Full url location of the file
        try:
            # Access the file via its url
            response = requests.get(url)
            response.raise_for_status()  # Will raise an HTTPError for bad requests (400+)
            data = StringIO(response.text)

            print(f'Attempting to retrieve file {i}/100')
            df = pd.read_csv(data, low_memory=False) # Read the data
            dfs.append(df) # Add the data to the new dataframe
            print(f'file {i} retrieved successfully')

        # Error handling
        except requests.exceptions.HTTPError as err:
            print(f"Failed to retrieve {file_name}: {err}")
        except Exception as e:
            print(f"An error occurred: {e}")

    combined_df = pd.concat(dfs, ignore_index=True) # Concatenate into a single dataframe
    return combined_df # Return the concatenated dataframe

def create_dataset(INPUT_FILES):
    """
    - Initialises data retrieval via the specified method ('local' or 'github').
    - Initialises data processing.
    - Creates and saves each file.

    """ 
    print('Initialising file retrieval...')

    # Retrieve data based on the selected method ('local' or 'github')
    if INPUT_FILES == 'local':
        df = get_local_files()
    elif INPUT_FILES == 'github':
        df = get_github_files()

    print('Files retrieved.')

    # Specify the two output file variants
    dimensions = ['2D', '3D']
        
    print('Initialising data processing...')

    # Process and save a file for both 2D and 3D
    for dimension in dimensions:
        df_copy = df.copy() # Create a copy of the dataset so not to affect the second iteration
        processed_df = process_dataframe(df_copy, dimension) # Process the data
        print(f'Saving {dimension} .csv file...')
        file_name = f"{OUTPUT_DIRECTORY}{dimension}_stress_field.csv" # Name the file
        processed_df.to_csv(file_name, index=False) # Save the file

        print(f'{dimension} dataset created and saved.')

    print('End of programme.')


#---------------------------------------------------------------------------------------------------------------------------------------------------------

create_dataset(INPUT_FILES) # Run the program