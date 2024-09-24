import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import os
import sys
from matplotlib.ticker import FuncFormatter
import time
from datetime import datetime, timedelta
import matplotlib.dates as mdates
#%%
def read_txt(path, parent_path, file_names, separation):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for name in file_names:
        for file in files:
            if name in file:
                with open(os.path.join(path, file)) as f:
                    df = pd.read_table(f, sep = separation)
                    new_dict[name] = df
    
    return new_dict

def read_txt_acsm(path, parent_path, file_names, separation):
    new_dict = {}
    data_dict = read_txt(path, parent_path, file_names, separation)

    for key in data_dict.keys():
        df = data_dict[key]
        df.columns = ['Time', 'org_conc']
        df['Time'] = format_timestamps(df['Time'], "%Y/%m/%d %H:%M:%S")
        df['Time'] = df['Time'] + pd.Timedelta(hours=2)
        new_dict[key] = df
    
    return new_dict

def format_timestamps(timestamps, old_format, new_format="%d/%m/%Y %H:%M:%S.%f"):
    new_timestamps = []
    for timestamp in timestamps:
        old_datetime = datetime.strptime(timestamp, old_format)
        new_datetime = old_datetime.strftime(new_format)
        new_timestamps.append(new_datetime)
    return pd.to_datetime(new_timestamps, format="%d/%m/%Y %H:%M:%S.%f")

def read_data(path, parent_path, time_label):
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        name = file.split('.')[0]
        with open(os.path.join(path, file)) as f:
            df = pd.read_csv(f, sep = ';', decimal=',')
            df[time_label] = format_timestamps(df[time_label], "%m/%d/%Y %H:%M:%S.%f")
            df = df.dropna()

            df['PAH total'] = pd.to_numeric(df['PAH total'], errors = 'coerce')

            df['Time'] = df[time_label] + pd.Timedelta(hours=2)

        data_dict[name] = df
    
    return data_dict

def format_timestamps(timestamp_series):
    """Convert a timestamp series to datetime format, detecting format automatically."""
    return pd.to_datetime(timestamp_series, errors='coerce', dayfirst=True)

def read_LCS_data(path, parent_path, time_label):
    """Read LCS data from CSV files in the specified path."""
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if file.lower().endswith('.csv'):
            name = file.split('.')[0]
            try:
                with open(os.path.join(path, file)) as f:
                    # Read the CSV file
                    df = pd.read_csv(f, sep=';', decimal=',') if file.endswith('.CSV') else pd.read_csv(f)

                    # Debug: Print the columns in the DataFrame
                    print(f"Columns in {file}: {df.columns.tolist()}")
                    
                    # Check if the time_label exists in the DataFrame
                    if time_label not in df.columns:
                        print(f"Column '{time_label}' not found in file {file}. Available columns: {df.columns.tolist()}")
                        continue  # Skip to the next file
                    
                    # Process the timestamp column
                    df[time_label] = format_timestamps(df[time_label])
                    
                    # Drop NA values
                    df = df.dropna()
                    
                    # Convert additional columns to numeric if they exist
                    if 'SPS30_PM2.5' in df.columns:
                        df['SPS30_PM2.5'] = pd.to_numeric(df['SPS30_PM2.5'], errors='coerce')
                    
                    # Create a timestamp with timezone adjustment
                    df['timestamp'] = df[time_label] + pd.Timedelta(hours=2)

                    # Store the DataFrame in the data_dict with its name as key
                    data_dict[name] = df
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return data_dict


def read_LCS_weather_data(path, parent_path, time_label):
    """Read LCS data from CSV files in the specified path."""
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if file.lower().endswith('.csv'):
            name = file.split('.')[0]
            try:
                with open(os.path.join(path, file)) as f:
                    # Read the CSV file
                    df = pd.read_csv(f, sep=';', decimal='.') if file.endswith('.CSV') else pd.read_csv(f)

                    # Debug: Print the columns in the DataFrame
                    print(f"Columns in {file}: {df.columns.tolist()}")
                    
                    # Check if the time_label exists in the DataFrame
                    if time_label not in df.columns:
                        print(f"Column '{time_label}' not found in file {file}. Available columns: {df.columns.tolist()}")
                        continue  # Skip to the next file
                    
                    # Process the timestamp column
                    df[time_label] = format_timestamps(df[time_label])
                    
                    # Drop NA values
                    df = df.dropna()
                    
                    # Convert additional columns to numeric if they exist
                    if 'Conc' in df.columns:
                        df['Conc'] = pd.to_numeric(df['Conc'], errors='coerce')
                    
                    # Create a timestamp with timezone adjustment
                    df['timestamp'] = df[time_label] + pd.Timedelta(hours=2)

                    # Store the DataFrame in the data_dict with its name as key
                    data_dict[name] = df
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return data_dict



# Define paths and parameters
parent_LCS = '../../../'
path_LCS = 'L:/PG-Nanoteknologi/PROJEKTER/2024 Laura og Nan/pilot kbh hovedbanegaard and noerregade/rawdata/particle/LCS/'
names_LCS = ['LCS0076', 'LCS0104']

# Read the LCS data
data = read_LCS_data(path_LCS, parent_LCS, 'timestamp')

# Debug: Print available keys in the data
print("Available keys in data:", data.keys())

# Display the data for 'LCS0076' if it exists
if 'LCS0076' in data:
    display(data['LCS0076'])
else:
    print("LCS0076 data is not available.")




def plot_Conc_ACSM(ax, fig, data_dict, dict_keys, concentration, ylabel):
    for i, dict_key in enumerate(dict_keys):
        df = data_dict[dict_key]
        #for j, key in enumerate(df.keys()[1:]):
        ax[i].plot(df['Time'], df[concentration], lw = 1)
    

        # Set the x-axis major formatter to a date format
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
        ax[i].xaxis.set_major_locator(mdates.AutoDateLocator())

        # Rotate and format date labels
        plt.setp(ax[i].xaxis.get_majorticklabels()) #, rotation=45, ha='right')

        

        ax[i].tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
        ax[i].set_title(dict_key, fontsize = 9)
    fig.supxlabel('Time', fontsize = 10)
    fig.supylabel(ylabel, fontsize = 10)
    


def plot_LCS(ax, fig, data_dict, dict_keys, start_time, end_time, concentration, ylabel):
    # Convert start_time and end_time to datetime objects if they are strings
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    for i, dict_key in enumerate(dict_keys):
        # Extract the timestamp and concentration data
        time = pd.to_datetime(data_dict[dict_key]['timestamp'])  # Ensure this is datetime
        Conc = np.array(data_dict[dict_key][concentration])

        # Create a filter for the time interval
        time_filter = (time >= start_time) & (time <= end_time)

        # Apply the time filter to both time and concentration data
        time_filtered = time[time_filter]
        Conc_filtered = Conc[time_filter]

        # Plot the filtered data
        ax[i].plot(time_filtered, Conc_filtered, lw=1)

        # Set x-axis major formatter to a date format
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Set the locator for the x-axis
        ax[i].xaxis.set_major_locator(mdates.AutoDateLocator())

        # Rotate and format date labels
        plt.setp(ax[i].xaxis.get_majorticklabels())

        # Set tick parameters and title
        ax[i].tick_params(axis='both', which='major', direction='out', bottom=True, left=True, labelsize=8)
        ax[i].set_title(dict_key, fontsize=9)
        ax[i].set_ylabel(ylabel[i], fontsize = 8)
    # Add common x and y labels for the figure
    fig.supxlabel('Time', fontsize=10)
    

