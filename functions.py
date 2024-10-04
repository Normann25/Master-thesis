import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import os
import sys
from matplotlib.ticker import FuncFormatter
import time
from datetime import datetime, timedelta
from iminuit import Minuit
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

def format_timestamps(timestamps, old_format, new_format):
    new_timestamps = []
    for timestamp in timestamps:
        old_datetime = datetime.strptime(str(timestamp), old_format)
        new_datetime = old_datetime.strftime(new_format)
        new_timestamps.append(new_datetime)
    return pd.to_datetime(new_timestamps, format = new_format)

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

def format_timestamps_v2(timestamp_series):
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
        if '.CSV' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep=';', decimal=',')
        if '.csv' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep=';', decimal='.')

                    # Debug: Print the columns in the DataFrame
                    # print(f"Columns in {file}: {df.columns.tolist()}")
                    
                    # # Check if the time_label exists in the DataFrame
                    # if time_label not in df.columns:
                    #     print(f"Column '{time_label}' not found in file {file}. Available columns: {df.columns.tolist()}")
                    #     continue  # Skip to the next file
                    
        # Process the timestamp column
        df[time_label] = format_timestamps(df[time_label], '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M')
        
        # Drop NA values
        # df = df.dropna()
        
        # Convert additional columns to numeric if they exist
        if 'SPS30_PM2.5' in df.columns:
            df['SPS30_PM2.5'] = pd.to_numeric(df['SPS30_PM2.5'], errors='coerce')
        
        # Create a timestamp with timezone adjustment
        df['timestamp'] = df[time_label] + pd.Timedelta(hours=2)

        # Store the DataFrame in the data_dict with its name as key
        data_dict[name] = df

    return data_dict


def read_LCS_weather_data(path, parent_path, time_label):
    """Read LCS data from CSV files in the specified path."""
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if '.CSV' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep=';', decimal=',')
        if '.csv' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep=';', decimal='.')

            # # Debug: Print the columns in the DataFrame
            # print(f"Columns in {file}: {df.columns.tolist()}")
            
            # # Check if the time_label exists in the DataFrame
            # if time_label not in df.columns:
            #     print(f"Column '{time_label}' not found in file {file}. Available columns: {df.columns.tolist()}")
            #     continue  # Skip to the next file
            
            # Process the timestamp column
            df[time_label] = format_timestamps(df[time_label])
            
            # Drop NA values
            # df = df.dropna()
            
            # Convert additional columns to numeric if they exist
            if 'Conc' in df.columns:
                df['Conc'] = pd.to_numeric(df['Conc'], errors='coerce')
            
            # Create a timestamp with timezone adjustment
            df['timestamp'] = df[time_label] + pd.Timedelta(hours=2)

            # Store the DataFrame in the data_dict with its name as key
            data_dict[name] = df

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
    
def plot_LCS_single(ax, data_dict, dict_key, start_time, end_time, concentration, ylabel):
    # Convert start_time and end_time to datetime objects if they are strings
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Extract the timestamp and concentration data
    time = pd.to_datetime(data_dict[dict_key]['timestamp'])  # Ensure this is datetime
    Conc = np.array(data_dict[dict_key][concentration])

    # Create a filter for the time interval
    time_filter = (time >= start_time) & (time <= end_time)

    # Apply the time filter to both time and concentration data
    time_filtered = time[time_filter]
    Conc_filtered = Conc[time_filter]

    # Plot the filtered data
    ax.plot(time_filtered, Conc_filtered, lw=1)

    # Set x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Set the locator for the x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels())

    # Set tick parameters and title
    ax.tick_params(axis='both', which='major', direction='out', bottom=True, left=True, labelsize=8)
    ax.set_title(dict_key, fontsize=9)
    ax.set_ylabel(ylabel, fontsize = 8)

def plot_LCS(ax, fig, data_dict, dict_keys, start_time, end_time, concentration, ylabel):
    for i, dict_key in enumerate(dict_keys):
        plot_LCS_single(ax[i], data_dict, dict_key, start_time, end_time, concentration, ylabel[i])

    # Add common x and y labels for the figure
    fig.supxlabel('Time', fontsize=10)

def plot_LCS_WS(ax, fig, data_dict, start_time, end_time, titles):
    for i, st in enumerate(start_time):
        plot_LCS_single(ax[i], data_dict, 'LCS0076', st, end_time[i], 'SPS30_PM2.5', 'PM$_{2.5}$ / $\mu$g/m$^{3}$')
        plot_LCS_single(ax[i], data_dict, 'LCS0104', st, end_time[i], 'SPS30_PM2.5', 'PM$_{2.5}$ / $\mu$g/m$^{3}$')
        plot_LCS_single(ax[i], data_dict, 'PM25', st, end_time[i], 'Conc', 'PM$_{2.5}$ / $\mu$g/m$^{3}$')
        # handles, = ax[i].get_legend_handles_labels()
        ax[i].legend(labels = ['LCS 0076', 'LCS 0104', 'Weather station'], frameon = False, fontsize = 8)
        ax[i].set_title(titles[i])

    # Add common x and y labels for the figure
    fig.supxlabel('Time', fontsize=10)

def get_mean_conc(data, dict_keys, timelabel, date, timestamps, concentration, path):
    pd.options.mode.chained_assignment = None  # suppress warnings
    
    idx_array = []
    for key in dict_keys:
        idx_ts = np.zeros(len(timestamps))
        for j, ts in enumerate(timestamps):
            for k, time in enumerate(data[key][timelabel]):
                full_time = str(time)
                search_time = date + ts
                if search_time in full_time:
                    idx_ts[j] = k  # store the first match
                    break  # stop after finding the first match
        idx_array.append(idx_ts)
        print(f"Indexes for {key}: {idx_ts}")

    mean_df = pd.DataFrame()
    for i, key in enumerate(dict_keys):
        mean_conc = []
        time_start = []
        time_end = []
        for j, idx in enumerate(idx_array[i][::2]):
            idx_start = int(idx)
            idx_end = int(idx_array[i][j * 2 + 1])
            
            if idx_start < len(data[key]) and idx_end <= len(data[key]):
                new_df = data[key].iloc[idx_start:idx_end, :]
                time_start.append(data[key][timelabel].iloc[idx_start])
                time_end.append(data[key][timelabel].iloc[idx_end])
                mean = new_df[concentration].mean()
                mean_conc.append(mean)
            else:
                print(f"Index out of bounds for {key}: start={idx_start}, end={idx_end}")

        mean_df[key + ' time start'] = time_start
        mean_df[key + ' time end'] = time_end
        mean_df[key] = mean_conc

    mean_df.to_csv(path)

    return mean_df

def linear_fit(x, y, a_guess, b_guess):

    Npoints = len(y)

    def fit_func(x, a, b):
        return b + (a * x)

    def least_squares(a, b) :
        y_fit = fit_func(x, a, b)
        squares = np.sum((y - y_fit)**2)
        return squares
    least_squares.errordef = 1.0    # Chi2 definition (for Minuit)

    # Here we let Minuit know, what to minimise, how, and with what starting parameters:   
    minuit = Minuit(least_squares, a = a_guess, b = b_guess)

    # Perform the actual fit:
    minuit.migrad();

    # Extract the fitting parameters:
    a_fit = minuit.values['a']
    b_fit = minuit.values['b']

    Nvar = 2                     # Number of variables 
    Ndof_fit = Npoints - Nvar    # Number of degrees of freedom = Number of data points - Number of variables

    # Get the minimal value obtained for the quantity to be minimised (here the Chi2)
    squares_fit = minuit.fval                          # The chi2 value

    # Calculate R2
    def simple_model(b):
        return b

    def least_squares_simple(b) :
        y_fit = simple_model(b)
        squares = np.sum((y - y_fit)**2)
        return squares
    least_squares_simple.errordef = 1.0    # Chi2 definition (for Minuit)

    # Here we let Minuit know, what to minimise, how, and with what starting parameters:   
    minuit_simple = Minuit(least_squares_simple, b = b_guess)

    # Perform the actual fit:
    minuit_simple.migrad();

    # Get the minimal value obtained for the quantity to be minimised (here the Chi2)
    squares_simple = minuit_simple.fval                          # The chi2 value

    R2 = 1 - (squares_fit / squares_simple)

    # Print the fitted parameters
    print(f"Fit: a={a_fit:6.6f}  b={b_fit:5.3f}  R2={R2:6.6f}")
    
    return a_fit, b_fit, squares_fit, Ndof_fit, R2

def plot_reference_LCS(ax, data_dict, dict_keys, start_time, end_time, concentration, axis_labels):

    # Convert start_time and end_time to datetime objects if they are strings
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Extract time and concentration data for both datasets
    time_1 = pd.to_datetime(data_dict[dict_keys[0]]['timestamp'])
    conc_1 = np.array(data_dict[dict_keys[0]][concentration[0]])

    time_2 = pd.to_datetime(data_dict[dict_keys[1]]['timestamp'])
    conc_2 = np.array(data_dict[dict_keys[1]][concentration[1]])

    # Apply the time filter to both datasets
    time_filter_1 = (time_1 >= start_time) & (time_1 <= end_time)
    time_filter_2 = (time_2 >= start_time) & (time_2 <= end_time)

    filtered_time_1 = time_1[time_filter_1]
    filtered_conc_1 = conc_1[time_filter_1]

    filtered_time_2 = time_2[time_filter_2]
    filtered_conc_2 = conc_2[time_filter_2]

    # Create DataFrames to align both datasets by timestamp
    df_1 = pd.DataFrame({'timestamp': filtered_time_1, dict_keys[0]: filtered_conc_1})
    df_2 = pd.DataFrame({'timestamp': filtered_time_2, dict_keys[1]: filtered_conc_2})

    # Now let's reapply the merging logic and see if it works
    merged_df = pd.merge(
    data['PM25'][['timestamp', 'Conc']],
    data['LCS0076'][['timestamp', 'SPS30_PM2.5']],
    on='timestamp', 
    how='inner'
    )

    # Plot a scatter plot of the two concentrations
    ax.scatter(merged_df[concentration[0]], merged_df[concentration[1]], s=10, c='blue', label=f'{dict_keys[0]} vs {dict_keys[1]}')

    x_plot = np.linspace(min(merged_df[concentration[0]]), max(merged_df[concentration[0]]), 100)
    a, b, squares, ndof, R2 = linear_fit(merged_df[concentration[0]], merged_df[concentration[1]], 1, 1)
    y_fit = a*x_plot + b

    ax.plot(x_plot, y_fit, label = 'Fit', color = 'k', lw = 1.2)

    # Set labels and title for the scatter plot
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', bottom = True, left = True)
    ax.set_xlabel(axis_labels[0], fontsize=8)
    ax.set_ylabel(axis_labels[1], fontsize=8)

    ax.legend(frameon = False, fontsize = 8)