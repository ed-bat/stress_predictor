"""
data_visualisation.py

File used to generate data plots and statistics from the full datasets
"""



# Adjust the following directory to the location where the stress files are stored
FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/" 

# Where the outputs from this script will be saved
OUTPUT_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/data_visualisation/" 



#----------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import FuncFormatter
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import FunctionTransformer


def get_contour_data(target_travel_length, target_travel_speed, target_heat_input): 
    file_path = f"{FILE_DIRECTORY}3D_stress_field.csv"
    df = pd.read_csv(file_path)
    
    # Apply filters using the target values
    data = df[(df['travel_length'] == target_travel_length) & 
                        (df['travel_speed'] == target_travel_speed) & 
                        (df['heat_input'] == target_heat_input)]

    return data


def plot_contours(target_travel_length, target_travel_speed, target_heat_input):
    contour_data = get_contour_data(target_travel_length, target_travel_speed, target_heat_input)
    dimension_pairs = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
    aspect_ratios = {'XY': 0.8, 'XZ': 1.3, 'YZ': 0.8}  # Specified aspect ratios for each plot

    for dimensions in dimension_pairs:
        dimension1, dimension2 = dimensions
        if dimension1 == 'X' and dimension2 == 'Y':
            plane_data = contour_data[np.isclose(contour_data['Z'], 0, atol=1e-3)]
            aspect_ratio = aspect_ratios['XY']
        elif dimension1 == 'X' and dimension2 == 'Z':
            plane_data = contour_data[np.isclose(contour_data['Y'], 0, atol=1e-3)]
            aspect_ratio = aspect_ratios['XZ']
        elif dimension1 == 'Y' and dimension2 == 'Z':
            plane_data = contour_data[np.isclose(contour_data['X'], 0, atol=1e-3)]
            aspect_ratio = aspect_ratios['YZ']

        # Create grid and interpolate for specified plane
        x_label, y_label = dimension1, dimension2
        x = np.linspace(plane_data[x_label].min()*1000, plane_data[x_label].max()*1000, 50)
        y = np.linspace(plane_data[y_label].min()*1000, plane_data[y_label].max()*1000, 50)
        X, Y = np.meshgrid(x, y)
        stress = griddata((plane_data[x_label]*1000, plane_data[y_label]*1000), plane_data['von_mises'] / 1e6, (X, Y), method='linear')

        plt.figure(figsize=(12, 6)) 
        contour = plt.contourf(X, Y, stress, levels=20, cmap='inferno')
        cbar = plt.colorbar(contour, format='%.2f')
        cbar.set_label('von Mises stress (MPa)')
        plt.title(f'von Mises Stress Distribution on the {dimension1}-{dimension2} Plane\nWeld length: {target_travel_length}mm, Weld speed: {target_travel_speed/1000}mm/s, Heat input: {target_heat_input/1000}kW')
        plt.xlabel(f'{x_label} (mm)')
        plt.ylabel(f'{y_label} (mm)')
        plt.gca().set_aspect(aspect_ratio, adjustable='box')  # Set the specified aspect ratio

        # Invert Y-axis for the 'X-Z' and 'Y-Z' plots only
        if dimension1 != 'X' or dimension2 != 'Y':
            plt.gca().invert_yaxis()

        filename = f"{OUTPUT_DIRECTORY}{dimension1}-{dimension2}_von_mises_stress_contour_{target_travel_length}_{target_travel_speed}_{target_heat_input}.jpeg"
        plt.savefig(filename, format='jpeg', dpi=150)
        plt.close()


def save_summary_statistics():
    file_path = f"{FILE_DIRECTORY}3D_stress_field.csv"
    data = pd.read_csv(file_path)
    
    # Compute summary statistics
    summary_stats = data.describe()
    
    # Save the summary statistics to a CSV file
    summary_stats.to_csv(f"{OUTPUT_DIRECTORY}data_statistics.csv")
    
    print(f"Summary statistics saved.")


def save_histograms():
    # Load the data
    data = pd.read_csv(f"{FILE_DIRECTORY}3D_stress_field.csv")

    # columns_to_drop = ['travel_length', 'travel_speed', 'heat_input', 'X', 'Y', 'Z']
    columns_to_drop = ['S11', 'S22', 'S33', 'S12', 'S23', 'S13', 'von_mises']

    data.drop(columns=columns_to_drop, inplace=True)
    
    plt.figure(figsize=(20, 16))  # Increased figure size for better visibility
    
    # Create histograms
    ax = data.hist(bins=100, grid=True, figsize=(20, 16), color='navy', zorder=2)
    
    # Set the style
    plt.style.use('ggplot')

    # Customise title
    for axis in ax.flatten():
        axis.set_title(axis.get_title(), fontsize=16, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    # Save the plot to a JPEG file
    plt.savefig(f"{OUTPUT_DIRECTORY}stress_histograms.jpg", format='jpeg', dpi=150)
    plt.close()
    print("Histograms saved.")


#-------------------------------------------------------------------------------------------------------------------------

plot_contours(target_travel_length=80, target_travel_speed=2000, target_heat_input=3000)
save_summary_statistics()
save_histograms()