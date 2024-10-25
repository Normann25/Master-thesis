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

def plot_LCS_single(ax, data_dict, dict_key, timelabel, start_time, end_time, concentration, ylabel):
    # Convert start_time and end_time to datetime objects if they are strings
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Extract the timestamp and concentration data
    time = pd.to_datetime(data_dict[dict_key][timelabel])  # Ensure this is datetime
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
        plot_LCS_single(ax[i], data_dict, dict_key, 'timestamp', start_time, end_time, concentration, ylabel[i])

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

def OPS_single_timeseries(ax, df, colors, timestamps):

    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    time = pd.to_datetime(df['Time'])
    conc = np.array(df['Total Conc'])
    conc = pd.to_numeric(conc, errors='coerce')

    time_filter = (time >= start_time) & (time <= end_time)

    filtered_time = time[time_filter]
    filtered_conc = conc[time_filter]

    new_df = pd.DataFrame({'Time': filtered_time, 'Total Conc': filtered_conc})

    for key in df.keys()[1:18]:
        conc = np.array(df[key])
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]

        new_df[key] = filtered_conc
        # ax.plot(df['Time'], df[key], label = key, color = colors[i+1], lw = 1)

    # Bin plot
    ax.plot(new_df['Time'], new_df['Total Conc'], label = 'Total', zorder = 10, color = colors[0], lw = 1)
    ax.plot(new_df['Time'], new_df.iloc[:,2:5].sum(axis = 1), color = colors[1], label = '0.300 - 0.465 $\mu$m')
    ax.plot(new_df['Time'], new_df.iloc[:,5:8].sum(axis = 1), color = colors[2], label = '0.465 - 0.897 $\mu$m')
    ax.plot(new_df['Time'], new_df.iloc[:,8:11].sum(axis = 1), color = colors[3], label = '0.897 - 1.732 $\mu$m')
    ax.plot(new_df['Time'], new_df.iloc[:,11:13].sum(axis = 1), color = colors[4], label = '1.732 - 2.685 $\mu$m')
    ax.plot(new_df['Time'], new_df.iloc[:,13:15].sum(axis = 1), color = colors[5], label = '2.685 - 4.162 $\mu$m')
    ax.plot(new_df['Time'], new_df.iloc[:,15:].sum(axis = 1), color = colors[6], label = '4.162 - 10.000 $\mu$m')

    # Set the x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels(), size = 8)
    plt.setp(ax.yaxis.get_majorticklabels(), size = 8)

    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (4,4))

    ax.set_xlabel('Time [HH:MM]', fontsize = 9)
    ax.set_ylabel('Concentration / #/cm$^{3}$', fontsize = 9)

    ax.legend(frameon = False, fontsize = 8, ncol = 1)


def NanoScan_single_timeseries(ax, df, ncol):
    
    ax.plot(df['Time'], df['Total Conc'], label = 'Total', zorder = 10)

    for key in df.keys()[3:16]:
        ax.plot(df['Time'], df[key], label = key + ' nm', lw = 1)

    # Set the x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels(), size = 8)

    ax.set_xlabel('Time [HH:MM]', fontsize = 9)
    ax.set_ylabel('Concentration / #/cm$^{3}$', fontsize = 9)

    ax.legend(frameon = False, fontsize = 8, ncol = ncol)

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
    # try:
    # Fill the generated mesh with particle concentration data
    p = ax.pcolormesh(x, y, data, cmap='jet',vmin=y_min, vmax=y_max,shading='flat')
    
    # except TypeError:
    #     data = data[:171, :12]  # Adjust data dimensions to (171, 12)

    #     # Fill the generated mesh with particle concentration data
    #     p = ax.pcolormesh(x, y, data, cmap='jet',vmin=y_min, vmax=y_max,shading='flat')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time, HH:MM")
    plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    ax.set_yscale("log")
    ax.set_ylabel("Dp, nm")
    
    # Insert coloarbar and label it
    col = fig.colorbar(p, ax=ax)
    if datatype == "number":
        col.set_label('dN, cm$^{-3}$')
    elif datatype == "mass":
        col.set_label('dm, $\mu$g/m$^{3}$')
    elif datatype == "normed":
        col.set_label('dN/dlogDp, cm$^{-3}$')
    
    # Set ticks on the plot to be longer
    ax.tick_params(axis="y",which="both",direction='out')
