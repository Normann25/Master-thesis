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
from iminuit import Minuit
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

def read_txt_acsm(path, parent_path, file_names, separation, hour):
    new_dict = {}
    data_dict = read_txt(path, parent_path, file_names, separation, None)

    for key in data_dict.keys():
        df = data_dict[key]
        df.columns = ['Time', 'org_conc']
        df['Time'] = format_timestamps(df['Time'], "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S.%f")
        df['Time'] = df['Time'] + pd.Timedelta(hours=hour)
        new_dict[key] = df
    
    return new_dict

def format_timestamps(timestamps, old_format, new_format):
    new_timestamps = []
    for timestamp in timestamps:
        old_datetime = datetime.strptime(timestamp, old_format)
        new_datetime = old_datetime.strftime(new_format)
        new_timestamps.append(new_datetime)
    return pd.to_datetime(new_timestamps, format=new_format)

def read_data(path, parent_path, time_label, hour):
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        name = file.split('.')[0]
        with open(os.path.join(path, file)) as f:
            df = pd.read_csv(f, sep = ';', decimal=',')
            df[time_label] = format_timestamps(df[time_label], "%m/%d/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S.%f")
            df = df.dropna()

            df['PAH total'] = pd.to_numeric(df['PAH total'], errors = 'coerce')

            df['Time'] = df[time_label] + pd.Timedelta(hours=hour)

        data_dict[name] = df
    
    return data_dict

def read_csv_BC(path, parent_path):
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    
    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if 'MA200' or 'MA300' in file:
            serial_number = linecache.getline(os.path.join(path, file), 2)
            serial_number = serial_number.split(',')[0]
            serial_number = serial_number.split('"')[1]
            name = file.split('.')[0]
            name = serial_number + '_' + name.split('_')[-1]

            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f)
                
                df['Time'] = df[['Date local (yyyy/MM/dd)', 'Time local (hh:mm:ss)']].agg(' '.join, axis=1)

                df['Time'] = format_timestamps(df['Time'], '%Y/%m/%d %H:%M:%S', "%d/%m/%Y %H:%M")

                # new_df = pd.DataFrame()
                # columns = ['Time', 'Sample temp (C)', 'Sample RH (%)', 'UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc', 'IR BCc']
                # for col in columns:
                #     new_df[col] = df[col]

                # new_df = new_df.dropna()
                for key in df.keys():
                    if 'BCc' in key:
                        df[key][df[key] < 0] = 0
                        df[key] = df[key] / 1000

            data_dict[name] = df

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

def read_LCS_data(path, parent_path, time_label, hour):
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
                    
        # Process the timestamp column
        df[time_label] = format_timestamps(df[time_label], '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M')
        
        # Convert additional columns to numeric if they exist
        if 'SPS30_PM2.5' in df.columns:
            df['SPS30_PM2.5'] = pd.to_numeric(df['SPS30_PM2.5'], errors='coerce')
        
        # Create a timestamp with timezone adjustment
        df['timestamp'] = df[time_label] + pd.Timedelta(hours=hour)

        # Store the DataFrame in the data_dict with its name as key
        data_dict[name] = df

    return data_dict

def read_SMPS(path, parent_path, hour):
    """Read LCS data from CSV files in the specified path."""
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if 'SCAN' in file:
            with open(os.path.join(path, file), 'r') as f:
                df = pd.read_table(f, sep = ',', skiprows = 8)

            df['Time'] = format_timestamps(df['Date Time'], "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S")
            df['Time'] = df['Time'] + pd.Timedelta(hours = hour[0])

            df['Date'] = pd.to_datetime(df['Date Time']).dt.date
            for date in df['Date'].unique():
                mask = df['Date'] == date
                new_df = df[mask].reset_index()
                data_dict[str(date) + '_NanoScan'] = new_df.drop('index', axis = 1)

        if 'SMPS' in file:
            separations = [',', '\t']
            name = file.split('.')[0]
            for separation in separations:
                try:
                    with open(os.path.join(path, file), 'r', encoding='ISO-8859-1') as f:
                        df = pd.read_table(f, sep = separation, skiprows = 17, header = None, index_col = 0)
                    
                    df = df.T

                    df['Time'] = df[['Date', 'Start Time']].agg(' '.join, axis=1)

                    df['Time'] = format_timestamps(df['Time'], '%m/%d/%y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                    df['Time'] = df['Time'] + pd.Timedelta(hours = hour[1])

                    data_dict[name] = df

                except KeyError:
                    print(f'Failed to read file with separation: {separation}')

    return data_dict

def read_OPS(path, parent_path, hr): # , V_chamber):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for file in files:
        if '.csv' in file:
            if 'OPS' in file:
                name = file.split('.')[0]
                name = name.split('_')[-1]
                start_date = linecache.getline(os.path.join(path, file), 8)
                start_date = start_date.split(',')[1]
                start_date = start_date.split('\n')[0]
                start_time = linecache.getline(os.path.join(path, file), 7)
                start_time = start_time.split(',')[1]
                start_time = start_time.split('\n')[0]
                start_time = start_date + ' ' + start_time
                old_time = datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S")
                new_time = old_time.strftime("%d/%m/%Y %H:%M:%S")

                DCT = linecache.getline(os.path.join(path, file), 34)
                DCT = float(DCT.split(',')[1])
                
                with open(os.path.join(path, file), 'r') as f:
                    df = pd.read_table(f, sep = ',', skiprows = 37)
                    df = df.drop(['Alarms', 'Errors', df.keys()[-1]], axis = 1)
                    df = df.dropna()
        
                Timestamps = []
                for time in df['Elapsed Time [s]']:
                    timestamp = pd.to_datetime(new_time, format="%d/%m/%Y %H:%M:%S") + pd.Timedelta(seconds = time) + pd.Timedelta(hours = hr)
                    Timestamps.append(timestamp)
                df['Time'] = Timestamps

                for key in df.keys()[1:18]:
                    C_i, ts, td = np.array(df[key]), np.zeros(len(df['Elapsed Time [s]'])) + df['Elapsed Time [s]'][0], np.array(df['Deadtime (s)'])

                    N_i = C_i / (16.67 * (ts - DCT * td))
                    df[key] = N_i # / V_chamber

                df['Total Conc']=df.iloc[:,1:18].sum(axis=1)

                new_dict[name] = df
        
        if 'APS' in file:
            separations = [',', '\t']
            name = file.split('.')[0]
            for separation in separations:
                try:
                    with open(os.path.join(path, file), 'r', encoding='ISO-8859-1') as f:
                        df = pd.read_table(f, sep = separation, skiprows = 6, header = None, index_col = 0)

                    df = df.T

                    df['Time'] = df[['Date', 'Start Time']].agg(' '.join, axis=1)

                    df['Time'] = format_timestamps(df['Time'], '%m/%d/%y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                    # df['Time'] = df['Time'] + pd.Timedelta(hours = hour)

                    df = df.drop(['Sample #', 'Date', 'Start Time'], axis = 1)

                    # if df['Aerodynamic Diameter'][1] == 'dN/dlogDp':
                    #     for key in df.keys()[2:53]:
                    #         df[key] = np.array(pd.to_numeric(df[key])) * np.log10(float(key))

                    new_dict[name] = df

                except KeyError:
                    print(f'Failed to read file with separation: {separation}')
        
    return new_dict

def read_partector(path, parent_path, names):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for name in names:
        for file in files:
            if name in file:
                start_date = linecache.getline(os.path.join(path, file), 5).split(' ')[1]
                start_time = linecache.getline(os.path.join(path, file), 5).split(' ')[-1]
                start_time = start_time.split('\n')[0]
                start_time = start_date + ' ' + start_time
                old_time = datetime.strptime(start_time, "%d.%m.%Y %H:%M:%S")
                new_time = old_time.strftime("%d/%m/%Y %H:%M:%S")
                
                with open(os.path.join(path, file), 'r') as f:
                    df = pd.read_table(f, sep = '\t', skiprows = 10)
                
                Timestamps = []
                for time in df['time']:
                    timestamp = pd.to_datetime(new_time, format="%d/%m/%Y %H:%M:%S") + pd.Timedelta(seconds = time)
                    Timestamps.append(timestamp)
                df['Time'] = Timestamps

                new_dict[name] = df
    
    return new_dict