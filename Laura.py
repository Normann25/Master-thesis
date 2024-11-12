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



def read_LCS_KU_data(path, time_label, names):

   files = os.listdir(path)
   data_dict = {}

   for name in names:
      for file in files:
         if name in file:
            with open(os.path.join(path, file)) as f:
               df = pd.read_table(f, sep = ',')

               display(df)

               # df[time_label] = pd.to_datetime(df[time_label])
               df[time_label] = format_timestamps(df[time_label], '%Y-%m-%d %H:%M:%S.%f', '%d/%m/%Y %H:%M')

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
                try:  
                    df = pd.read_excel(os.path.join(path, file))
                    df[df.keys()[0]] = format_timestamps(df[df.keys()[0]], '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M')
                    dates = []
                    for date in str(df[df.keys()[0]]).split(' ')[0].unique():
                        dates.append(date)
                    for date, sensor in zip(dates, df['Entity Name'].unique()):
                        mask = df['Entity Name'] == sensor
                        new_df = df[mask].reset_index()
                        df_name = str(date) + ' ' + str(sensor)
                        data_dict[df_name] = new_df.drop('index', axis = 1)
                except Exception as e:
                    print(f"An error occurred while processing {file}: {e}")

    return data_dict

