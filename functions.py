import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from matplotlib.ticker import FuncFormatter
import time
from datetime import datetime
import matplotlib.dates as mdates
import linecache
#%%
def read_txt(path, parent_path, file_names, separation, skip):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for name in file_names:
        for file in files:
            if name in file:
                with open(os.path.join(path, file)) as f:
                    df = pd.read_table(f, sep = separation, skiprows = skip)
                    new_dict[name] = df
    
    return new_dict

def read_txt_acsm(path, parent_path, file_names, separation):
    new_dict = {}
    data_dict = read_txt(path, parent_path, file_names, separation, None)

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

def read_csv_BC(path, parent_path):
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    
    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if '.csv' in file:
            name = file.split('.')[0]
            name = name.split('-')[1]
            name = name.split('_')[0] + '_' + name.split('_')[2]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f)
                
                df['Time'] = pd.to_timedelta(df['Time local (hh:mm:ss)']).astype('timedelta64[s]')  # .str.split().str[1]

                new_df = pd.DataFrame()
                columns = ['Time', 'Sample temp (C)', 'Sample RH (%)', 'UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc', 'IR BCc']
                for col in columns:
                    new_df[col] = df[col]

                new_df = new_df.dropna()
                for key in new_df.keys():
                    if 'BCc' in key:
                        new_df[key] = new_df[key] / 1000

            data_dict[name] = new_df

    return data_dict 

def read_discmini(path, parent_path, file_names, separation):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for name in file_names:
        for file in files:
            if name in file:
                start_date = linecache.getline(os.path.join(path, file), 3).split(' ')[-1]
                start_date = start_date.split(']')[0]
                start_time = linecache.getline(os.path.join(path, file), 4).split(' ')[-1]
                start_time = start_time.split(']')[0]
                start_time = start_date + ' ' + start_time
                old_time = datetime.strptime(start_time, "%Y.%m.%d %H:%M:%S")
                new_time = old_time.strftime("%d/%m/%Y %H:%M:%S")
                
                with open(os.path.join(path, file), 'r') as f:
                    df = pd.read_table(f, sep = separation, skiprows = 5)

                    Timestamps = []
                    for time in df['Time']:
                        timestamp = pd.to_datetime(new_time, format="%d/%m/%Y %H:%M:%S") + pd.Timedelta(seconds = time)
                        Timestamps.append(timestamp)
                    df['Time'] = Timestamps

                    new_df = pd.DataFrame()
                    columns = ['Time', 'Number', 'Size', 'LDSA', 'Filter', 'Diff']
                    for col in columns:
                        new_df[col] = df[col]
                    new_dict[name] = new_df
        
    return new_dict

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
    
        
