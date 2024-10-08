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
from calculations import *
#%%
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
    fig.supxlabel('Time [HH:MM]', fontsize = 10)
    fig.supylabel(ylabel, fontsize = 10)
    
def discmini_single_timeseries(ax, df, n):
    p1, = ax.plot(df['Time'], df['Number'], lw = 1, label = 'Number concentration', color = 'tab:blue')
    ax2 = ax.twinx()
    p2, = ax2.plot(df['Time'], df['LDSA'], lw = 0.5, ls = '--', label = 'LDSA', color = 'red')
    ax.set_zorder(ax2.get_zorder()+1)
    ax.set_frame_on(False)

    # Set the x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels())
    plt.setp(ax2.xaxis.get_majorticklabels())

    ylim = np.array(ax.get_ylim())
    ratio = ylim / np.sum(np.abs(ylim))
    scale = ratio / np.min(np.abs(ratio))
    scale = scale / n
    ax2.set_ylim(np.max(np.abs(ax2.get_ylim())) * scale)

    ax.tick_params(axis = 'y', labelcolor = p1.get_color(), labelsize = 8)
    ax.tick_params(axis = 'x', labelsize = 8)
    ax2.tick_params(axis = 'y', labelcolor = p2.get_color(), labelsize = 8)

    ax.legend(frameon = False, fontsize = 8, handles = [p1, p2])

    ax.set_xlabel('Time [HH:MM]', fontsize = 9)
    ax.set_ylabel('Concentration / #/cm$^{3}$', color = p1.get_color(), fontsize = 9)
    ax2.set_ylabel('LDSA / $\mu$m$^{2}$/cm$^{3}$', color = p2.get_color(), fontsize = 9)       

def discmini_multi_timeseries(ax, data, dict_keys, n, titles):
    for i, key in enumerate(dict_keys):
        df = data[key]
        discmini_single_timeseries(ax[i], df, n[i])
        ax[i].set_title(titles[i])

def ma200_single_timeseries(ax, df):
    ax.plot(df['Time'], df['IR BCc'])

    # Set the x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels(), size = 8)
    # formatter = FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
    # ax.xaxis.set_major_formatter(formatter)
    # ax.set_xticklabels(ax.get_xticklabels(), size = 8)

    ax.set_xlabel('Time [HH:MM]', fontsize = 9)
    ax.set_ylabel('Concentration / $\mu$g/m$^{3}$', fontsize = 9)

def ma200_multi_timeseries(ax, data, dict_keys):
    for i, key in enumerate(dict_keys):
        df = data[key]
        ma200_single_timeseries(ax[i], df)
        title = 'Bag ' + str(i+1) + ', MA ' + key.split('_')[0]
        ax[i].set_title(title)

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

def plot_reference(ax, data_dict, dict_keys, concentration, timelabel, x_plot, axis_labels):

    new_dict = {}
    for key in dict_keys:
        time = pd.to_datetime(data_dict[key][timelabel]).round('10s')
        conc = np.array(data_dict[key][concentration])
        df = pd.DataFrame({timelabel: time, key: conc})
        new_dict[key] = df

    # Merge the two dataframes
    merged = pd.DataFrame({timelabel: []})
    names = []
    for key in dict_keys:
        merged = pd.merge(merged, new_dict[key], on = timelabel, how = 'outer')
        names.append(key.split('_')[0])
    merged = merged.dropna()

    # Plot a scatter plot of the two concentrations
    ax.plot(x_plot, x_plot, color = 'grey', lw = 1, ls = '--')

    a, b, squares, ndof, R2 = linear_fit(merged[dict_keys[0]], merged[dict_keys[1]], 1, 0)
    y_fit = a*x_plot + b

    ax.plot(x_plot, y_fit, label = 'Fit', color = 'k', lw = 1.2)

    ax.scatter(merged[dict_keys[0]], merged[dict_keys[1]], s=10, c='blue', label=f'{names[0]} vs {names[1]}') 

    # Set labels and title for the scatter plot
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', bottom = True, left = True)
    ax.set_xlabel(axis_labels[0], fontsize=8)
    ax.set_ylabel(axis_labels[1], fontsize=8)
    ax.set(xlim = (min(x_plot), max(x_plot)), ylim = (min(x_plot), max(x_plot)))

    ax.legend(fontsize = 8)

def plot_reference_LCS(ax, data_dict, dict_keys, start_time, end_time, concentration, axis_labels):

    # Convert start_time and end_time to datetime objects if they are strings
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    new_dict = {}
    for i, key in enumerate(dict_keys):
        # Extract time and concentration data for both datasets
        time = pd.to_datetime(data_dict[key]['timestamp'])
        conc = np.array(data_dict[key][concentration[i]])

        # Apply the time filter to both datasets
        time_filter = (time >= start_time) & (time <= end_time)
        filtered_time = time[time_filter]
        filtered_conc = conc[time_filter]

        # Create DataFrames to align both datasets by timestamp
        df = pd.DataFrame({'timestamp': filtered_time, key: filtered_conc})
        new_dict[key] = df

    # Merge the two dataframes
    merged_df = pd.merge(new_dict[dict_keys[0]], new_dict[dict_keys[1]], on='timestamp', how='inner')

    # Plot a scatter plot of the two concentrations
    ax.scatter(merged_df[dict_keys[0]], merged_df[dict_keys[1]], s=10, c='blue', label=f'{dict_keys[0]} vs {dict_keys[1]}')

    x_plot = np.linspace(min(merged_df[dict_keys[0]]), max(merged_df[dict_keys[0]]), 100)
    a, b, squares, ndof, R2 = linear_fit(merged_df[dict_keys[0]], merged_df[dict_keys[1]], 1, 0)
    y_fit = a*x_plot + b

    ax.plot(x_plot, y_fit, label = 'Fit', color = 'k', lw = 1.2)

    # Set labels and title for the scatter plot
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', bottom = True, left = True)
    ax.set_xlabel(axis_labels[0], fontsize=8)
    ax.set_ylabel(axis_labels[1], fontsize=8)

    ax.legend(frameon = False, fontsize = 8)