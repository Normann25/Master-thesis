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
                    keys = ['PM5000S_2_PN0.3', 'PM5000S_2_PN0.5','PM5000S_2_PN1','PM5000S_2_PN2.5','PM5000S_2_PN5','PM5000S_2_PN10']
                    for key in keys: 
                        df[key] = pd.to_numeric(df[key]) / 1000
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
               keys = ['PN<1','PN<2.5','PN<5','PN<10']
               for key in keys: 
                   df[key] = pd.to_numeric(df[key])

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


def plot_timeseries(fig, ax, df, df_keys, bin_edges, datatype, timestamps):

    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    time = pd.to_datetime(df['Time'])

    time_filter = (time >= start_time) & (time <= end_time)

    filtered_time = np.array(time[time_filter])

    new_df = pd.DataFrame({'Time': filtered_time})

    for key in df_keys:
        conc = np.array(df[key])
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]

        new_df[key] = filtered_conc

    data = np.array(new_df[new_df.keys()[1:]])

    # Set the upper and/or lower limit of the color scale based on input
    y_min = np.nanmin(data)
    y_max = np.nanmax(data)

    # Generate an extra time bin, which is needed for the meshgrid
    dt = filtered_time[1]-filtered_time[0]
    new_time = filtered_time - dt
    new_time = np.append(new_time, new_time[-1]+dt)

    # generate 2d meshgrid for the x, y, and z data of the 3D color plot
    y, x = np.meshgrid(bin_edges, new_time)

    # Fill the generated mesh with particle concentration data
    p1 = ax[0].pcolormesh(x, y, data, cmap='jet',vmin=y_min, vmax=y_max,shading='flat')

    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=-45, ha="left")
    ax[0].set_xlabel("Time, HH:MM")
    plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Dp, nm")

    total_conc = new_df.iloc[:,1:].sum(axis=1)
    ax[1].plot(new_df['Time'], total_conc, lw = 1, color = 'r')

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=-45, ha="left")
    ax[1].set_xlabel("Time, HH:MM")
    plt.subplots_adjust(hspace=0.05)

    # Insert coloarbar and label it
    col = fig.colorbar(p1, ax=ax[0])
    if datatype == "number":
        col.set_label('dN, cm$^{-3}$')
        ax[1].set_ylabel('dN, cm$^{-3}$')
    elif datatype == "mass":
        col.set_label('dm, $\mu$g/m$^{3}$')
        ax[1].set_ylabel('dm, $\mu$g/m$^{3}$')
    elif datatype == "normed":
        col.set_label('dN/dlogDp, cm$^{-3}$')
        ax[1].set_ylabel('dN/dlogDp, cm$^{-3}$')

    # Set ticks on the plot to be longer
    ax[0].tick_params(axis="y",which="both",direction='out')
    ax[1].tick_params(axis="y",which="both",direction='out')




