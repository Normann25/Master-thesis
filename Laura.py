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


def read_CPC(path):
   
   files = os.listdir(path)
   new_dict = {}

   for file in files:
      name = file.split('.')[0]
      try:
         with open(os.path.join(path, file)) as f:
            df = pd.read_csv(f, sep = ',', skiprows=2)[:-2]
            df['Time'] = pd.to_datetime(df['Time'])
      
      except KeyError:
         with open(os.path.join(path, file)) as f:
            df = pd.read_csv(f, sep = ',', skiprows=17)[:-2]
            df['Time'] = pd.to_datetime(df['Time'])
      new_dict[name] = df
   return new_dict

def format_timestamps(timestamps, old_format, new_format):
    new_timestamps = []
    for timestamp in timestamps:
        old_datetime = datetime.strptime(timestamp, old_format)
        new_datetime = old_datetime.strftime(new_format)
        new_timestamps.append(new_datetime)
    return pd.to_datetime(new_timestamps, format=new_format)

# def plot_CPC_timeseries(ax, fig): 

def format_single_timestamp(column, from_format, to_format):
    timestamp = pd.to_datetime(column, format=from_format)
    return timestamp.strftime(to_format) if isinstance(timestamp, pd.Timestamp) else timestamp.dt.strftime(to_format)

def read_LCS_KU_data(path, time_label, names):
    
    files = os.listdir(path)
    data_dict = {}

    for name in names:
        for file in files:
            if name in file:
                with open(os.path.join(path, file)) as f:
                    df = pd.read_table(f, sep = ',')

                Timestamps = []
                for time in df[time_label]:
                    try:
                        timestamp = pd.to_datetime(time, format = '%Y-%m-%d %H:%M:%S')
                        Timestamps.append(timestamp)
                    except ValueError:
                        timestamp = format_single_timestamp(time, '%Y-%m-%d %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S')
                        Timestamps.append(timestamp)

                df[time_label] = Timestamps

                data_dict[name] =df 

    return data_dict


def read_LCS_data_OPC5000(path, time_label, names, sep):

   files = os.listdir(path)
   data_dict = {}

   for name in names:
      for file in files:
         if name in file:
            with open(os.path.join(path, file)) as f:
               df = pd.read_table(f, sep = sep)

               df[time_label] = format_timestamps(df[time_label], '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M')

            data_dict[name] =df 
   
   return data_dict



def format_timestamps(column, from_format, to_format):
    return pd.to_datetime(column, format=from_format).dt.strftime(to_format)

def read_LCS_data_LUND(path, time_label, names):
    files = os.listdir(path)
    data_dict = {}

    for name in names:
        for file in files:
            if name in file and file.endswith('.xlsx'): 

                # try:  
                df = pd.read_excel(os.path.join(path, file), decimal=',')

                dates = []
                for time in df[time_label]:
                    date = str(time).split(' ')[0]
                    dates.append(date)
                df['Date'] = dates

                df[time_label] = format_timestamps(df[time_label], '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M')

                for date in df['Date'].unique():
                    for sensor in df['Entity Name'].unique():
                        mask = df['Entity Name'] == sensor
                        new_df = df[mask].reset_index()
                        df_name = str(date) + ' ' + str(sensor)
                        data_dict[df_name] = new_df.drop('index', axis = 1)
                # except Exception as e:
                #     print(f"An error occurred while processing {file}: {e}")

    return data_dict

