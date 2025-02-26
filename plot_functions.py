import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import sys
from matplotlib.ticker import FuncFormatter
import time
from datetime import datetime
import matplotlib.dates as mdates
import linecache
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.append('..')
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

def plot_inset(ax, height, loc, bb2a, plot_width, xdata, ydata, width, bar, timeseries, timestamps):
    inset_ax = inset_axes(ax,
                            width = plot_width, # width = % of parent_bbox
                            height = height, # height : 1 inch
                            loc = loc,
                            bbox_to_anchor = bb2a,
                            bbox_transform = ax.transAxes) # placement in figure
    if bar:
        inset_ax.bar(xdata, ydata, width)
    
    if timeseries:
        start_time = pd.to_datetime(timestamps[0])
        end_time = pd.to_datetime(timestamps[1])
        time = pd.to_datetime(xdata)
        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = np.array(time[time_filter])
        conc = np.array(ydata)
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]
        inset_ax.plot(filtered_time, filtered_conc)

        # Set the x-axis major formatter to a date format
        inset_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
        inset_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Rotate and format date labels
        plt.setp(inset_ax.xaxis.get_majorticklabels(), size = 8)

    inset_ax.set(xlabel = None, ylabel = None)

    return inset_ax
    
def plot_MS(ax, df, conc, width, ttl):
    ax.bar(df['m/z'], df[conc], width)

    ax.set(xlabel = 'm/z', ylabel = 'Intensity', title = ttl)

def plot_MS_wInset(ax, data_dict, dict_keys, conc, height, loc, bb2a, widths, titles):
    for i, key in enumerate(dict_keys):
        df = data_dict[key]
        zero_mask = df[conc] >= 0
        df = df[zero_mask]

        plot_MS(ax[i], df, conc, widths[0], titles[i])
        
        ax[i].set_xlim(0, 300)

        mask = df['m/z'] >= 100
        inset_ax = plot_inset(ax[i], height, loc, bb2a, '60%', df['m/z'][mask], df[conc][mask], widths[1], True, False, None)

        inset_ax.set(xlim = (100, 300))


    
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

def ma_single_timeseries(ax, df, screening, timestamps, loc):
    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    time = pd.to_datetime(df['Time'])

    time_filter = (time >= start_time) & (time <= end_time)

    filtered_time = np.array(time[time_filter])

    conc_keys = ['UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc', 'IR BCc']
    colors = ['darkviolet', 'blue', 'green', 'red', 'k']

    if screening:
        conc = np.array(df['IR BCc'])
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]
        ax.plot(filtered_time, filtered_conc)
    
    else:
        for key, clr in zip(conc_keys, colors):
            conc = np.array(df[key])
            conc = pd.to_numeric(conc, errors='coerce')
            filtered_conc = conc[time_filter]
            ax.plot(filtered_time, filtered_conc, color = clr, label = key)
            ax.legend(frameon = False, fontsize = 8, loc = loc)

    # Set the x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels(), size = 8)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")

    ax.set_xlabel('Time [HH:MM]', fontsize = 9)
    ax.set_ylabel('Concentration / $\mu$g/m$^{3}$', fontsize = 9)

def ma_multi_timeseries(ax, data, dict_keys, screening, timestamps):
    for i, key in enumerate(dict_keys):
        df = data[key]
        ma_single_timeseries(ax[i], df, screening, timestamps)

        if screening:
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

def partector_single_timeseries(ax, df, timestamps, loc):
    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    time = pd.to_datetime(df['Time'])

    time_filter = (time >= start_time) & (time <= end_time)

    filtered_time = np.array(time[time_filter])

    conc = np.array(df['LDSA'])
    conc = pd.to_numeric(conc, errors='coerce')
    filtered_conc = conc[time_filter]
    ax.plot(filtered_time, filtered_conc, color = 'r', lw = 1)

    # Set the x-axis major formatter to a date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Set the locator for the x-axis (optional, depending on how you want to space the ticks)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Rotate and format date labels
    plt.setp(ax.xaxis.get_majorticklabels(), size = 8)

    ax.set_xlabel('Time [HH:MM]', fontsize = 9)
    ax.set_ylabel('LDSA / $\mu$m$^{2}$/cm$^{3}$', fontsize = 9)

def plot_timeseries(fig, ax, df, df_keys, bin_edges, datatype, timestamps, normed, cutpoint):

    def plot_heatmap(ax, df, time, bin_edges, cutpoint):
        data = np.array(df[df.keys()[1:]])

        if normed == False:
            dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
            data=data/dlogDp

        # Set the upper and/or lower limit of the color scale based on input
        y_min = np.nanmin(data)
        y_max = np.nanmax(data)

        # Generate an extra time bin, which is needed for the meshgrid
        dt = time[1]-time[0]
        new_time = time - dt
        new_time = np.append(new_time, new_time[-1]+dt)

        # generate 2d meshgrid for the x, y, and z data of the 3D color plot
        y, x = np.meshgrid(bin_edges, new_time)

        # Fill the generated mesh with particle concentration data
        p1 = ax.pcolormesh(x, y, data, cmap='jet',vmin=y_min, vmax=y_max,shading='flat')

        # ax.hlines(np.array([0.1, 2.5]), np.array([new_time[0], new_time[0]]), np.array([new_time[-1], new_time[-1]]), colors = 'white', linestyles = '--')

        if cutpoint != None:
            ax.hlines(cutpoint, new_time[0], new_time[-1], colors = 'white', linestyles = '--')
            # ax.text(cutpoint[0] - 0.1*cutpoint[0], new_time[10], cutpoint[1], fontsize = 9, color = 'white')
            # ax.text(cutpoint[0] + 0.1*cutpoint[0], new_time[10], cutpoint[2], fontsize = 9, color = 'white')

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        ax.set_xlabel("Time, HH:MM")
        plt.subplots_adjust(hspace=0.05)
            
        # Make the y-scal logarithmic and set a label
        ax.set_yscale("log")
        ax.set_ylabel("Dp, $\mu$m")
        return ax, p1
    
    def plot_total(ax, df):
        total_conc = df.iloc[:,1:].sum(axis=1)
        ax.plot(df['Time'], total_conc, lw = 1, color = 'r')

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        ax.set_xlabel("Time, HH:MM")
        plt.subplots_adjust(hspace=0.05)
        return ax
    
    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    if datatype == 'number and mass':
        df_number, df_mass = df[0], df[1]

        time = pd.to_datetime(df_number['Time'])

        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = np.array(time[time_filter])

        new_df_number, new_df_mass = pd.DataFrame({'Time': filtered_time}), pd.DataFrame({'Time': filtered_time})

        for key in df_keys:
            conc_number, conc_mass = np.array(df_number[key]), np.array(df_mass[key])
            conc_number, conc_mass = pd.to_numeric(conc_number, errors='coerce'), pd.to_numeric(conc_mass, errors='coerce')
            filtered_number, filtered_mass = conc_number[time_filter], conc_mass[time_filter]

            new_df_number[key], new_df_mass[key] = filtered_number, filtered_mass
        
        ax1, p1 = plot_heatmap(ax[0][0], new_df_number, filtered_time, bin_edges, cutpoint)
        ax2, p2 = plot_heatmap(ax[0][1], new_df_mass, filtered_time, bin_edges, cutpoint)

        ax3 = plot_total(ax[1][0], new_df_number)
        ax4 = plot_total(ax[1][1], new_df_mass)

        # Insert coloarbar and label it
        col1 = fig.colorbar(p1, ax=ax1)
        col2 = fig.colorbar(p2, ax=ax2)

        col1.set_label('dN/dlogDp, cm$^{-3}$')
        col2.set_label('dM/dlogDp, $\mu$g/m$^{3}$')
        
        ax3.set_ylabel('Total number conc., cm$^{-3}$')
        ax4.set_ylabel('Total mass conc., $\mu$g/m$^{3}$')

        # Set ticks on the plot to be longer
        ax1.tick_params(axis="y",which="both",direction='out')
        ax2.tick_params(axis="y",which="both",direction='out')

    else:
        time = pd.to_datetime(df['Time'])

        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = np.array(time[time_filter])

        new_df = pd.DataFrame({'Time': filtered_time})

        for key in df_keys:
            conc = np.array(df[key])
            conc = pd.to_numeric(conc, errors='coerce')
            filtered_conc = conc[time_filter]

            new_df[key] = filtered_conc

        ax1, p1 = plot_heatmap(ax[0], new_df, filtered_time, bin_edges, cutpoint)

        ax2 = plot_total(ax[1], new_df)

        # Insert coloarbar and label it
        col = fig.colorbar(p1, ax=ax1)
        if datatype == "number":
            col.set_label('dN/dlogDp, cm$^{-3}$')
            ax2.set_ylabel('Total concentration, cm$^{-3}$')
        elif datatype == "mass":
            col.set_label('dM/dlogDp, $\mu$g/m$^{3}$')
            ax2.set_ylabel('Total concentration, $\mu$g/m$^{3}$')

        # Set ticks on the plot to be longer
        ax1.tick_params(axis="y",which="both",direction='out')
        ax2.tick_params(axis="y",which="both",direction='out')

def plot_bin_mean(ax, timestamps, df_number, df_mass, df_keys, timelabel, bin_means, bin_edges, inst_error, cut_point, mass):
    mean_number, std_number, error_number = bin_mean(timestamps, df_number, df_keys, timelabel, inst_error)

    if bin_edges != None:
        dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
        mean_number=mean_number/dlogDp
        std_number=std_number/dlogDp
        error_number=error_number/dlogDp

    min_std_number = [m - std for m, std in zip(mean_number, std_number)]
    max_std_number = [m + std for m, std in zip(mean_number, std_number)]
    abs_error_number = [abs(error) for error in error_number]

    if cut_point == None:
        ax.fill_between(bin_means, min_std_number, max_std_number, alpha=0.2, color='tab:blue', linewidth=0)
        ax.errorbar(bin_means, mean_number, abs_error_number, ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5, color='tab:blue', lw = 1.2)
    else:
        df_number = pd.DataFrame({'Bin mean': bin_means, 'Concentration': mean_number, 'Std min': min_std_number, 'Std max': max_std_number, 'Error': abs_error_number})
        
        lower_cut = df_number['Bin mean'] < cut_point
        upper_cut = df_number['Bin mean'] > cut_point

        ax.fill_between(df_number['Bin mean'][lower_cut], df_number['Std min'][lower_cut], df_number['Std max'][lower_cut], alpha=0.2, color='tab:blue', linewidth=0)
        ax.errorbar(df_number['Bin mean'][lower_cut], df_number['Concentration'][lower_cut], df_number['Error'][lower_cut], ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5, color='tab:blue', lw = 1.2)

        ax.fill_between(df_number['Bin mean'][upper_cut], df_number['Std min'][upper_cut], df_number['Std max'][upper_cut], alpha=0.2, color='tab:blue', linewidth=0)
        ax.errorbar(df_number['Bin mean'][upper_cut], df_number['Concentration'][upper_cut], df_number['Error'][upper_cut], ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5, color='tab:blue', lw = 1.2)

    # Explicitly set ylabel color for primary axis
    ax.tick_params(axis = 'y', labelcolor='tab:blue')
    ax.set_ylabel('dN/dlogDp, cm$^{-3}$', color='tab:blue')

    ax.set(xlabel='Particle diameter / $\mu$m', xscale='log')

    if mass:
        mean_mass, std_mass, error_mass = bin_mean(timestamps, df_mass, df_keys, timelabel, inst_error)

        if bin_edges != None:
            dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
            mean_mass=mean_mass/dlogDp
            std_mass=std_mass/dlogDp
            error_mass=error_mass/dlogDp

        min_std_mass = [m - std for m, std in zip(mean_mass, std_mass)]
        max_std_mass = [m + std for m, std in zip(mean_mass, std_mass)]
        abs_error_mass = [abs(error) for error in error_mass]

        # Create a secondary y-axis for mass concentration
        ax2 = ax.twinx()
        
        # Plotting for the mass concentration
        if cut_point == None:
            ax2.fill_between(bin_means, min_std_mass, max_std_mass, alpha=0.2, color='red', linewidth=0)
            ax2.errorbar(bin_means, mean_mass, abs_error_mass, ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5, color='red', lw = 1.2)
        else:
            df_mass = pd.DataFrame({'Bin mean': bin_means, 'Concentration': mean_mass, 'Std min': min_std_mass, 'Std max': max_std_mass, 'Error': abs_error_mass})
            
            lower_cut = df_mass['Bin mean'] < cut_point
            upper_cut = df_mass['Bin mean'] > cut_point

            ax2.fill_between(df_mass['Bin mean'][lower_cut], df_mass['Std min'][lower_cut], df_mass['Std max'][lower_cut], alpha=0.2, color='red', linewidth=0)
            ax2.errorbar(df_mass['Bin mean'][lower_cut], df_mass['Concentration'][lower_cut], df_mass['Error'][lower_cut], ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5, color='red', lw = 1.2)

            ax2.fill_between(df_mass['Bin mean'][upper_cut], df_mass['Std min'][upper_cut], df_mass['Std max'][upper_cut], alpha=0.2, color='red', linewidth=0)
            ax2.errorbar(df_mass['Bin mean'][upper_cut], df_mass['Concentration'][upper_cut], df_mass['Error'][upper_cut], ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5, color='red', lw = 1.2)

        ax2.tick_params(axis = 'y', labelcolor='red')

        # Explicitly set ylabel color for secondary axis
        ax2.set_ylabel('dM/dlogDp, $\mu$g/m$^{3}$', color='red')  # Use axis_labels[2] for clarity
    
    else:
        ax2, mean_mass, error_mass = 0, 0, 0
    
    return mean_number, error_number, mean_mass, error_mass, ax, ax2

def plot_mean_all(timestamps, dict_number, dict_mass, dict_keys, df_keys, bins, bin_edges, inst_error, ymax):
    new_dict_keys = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
    mean_conc = {}
    axes = []
    figs = []

    fig1, ax1 = plt.subplots(2, 2, figsize = (8, 6))
    figs.append(fig1)

    for i, key in enumerate(dict_keys):
        if i == 0:
            ax = ax1[0][0]
        if i == 1:
            ax = ax1[0][1]
        if i == 2:
            ax = ax1[1][0]
        if i == 3:
            ax = ax1[1][1]

        number, error_number, mass, error_mass, ax_n, ax_m = plot_bin_mean(ax, timestamps[i], dict_number[key], dict_mass[key], df_keys, 'Time', bins, bin_edges, inst_error, None, True)
        axes.append([ax_n, ax_m])
        mean_conc[new_dict_keys[i]] = pd.DataFrame({'Diameter': np.array(bins), 'number': number, 'error number': error_number, 'mass': mass, 'error mass': error_mass})

        ax_n.set_ylim(0, ymax[0])
        ax_m.set_ylim(0, ymax[1])
        
        title = 'Experiment ' + str(i + 1)
        ax.set_title(title)

    sublabels = ['a', 'b', 'c', 'd']
    for ax, l in zip(ax1.flatten(), sublabels):
        ax.text(0.02, 0.92, l, transform = ax.transAxes, fontsize = 10)
    
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(1,2, figsize = (6.3, 3))  
    axes.append([ax2[0], ax2[1]])
    figs.append(fig2)
    labels = ['2 m/s - LM', '4 m/s - LM', '2 m/s - NAO', '4 m/s - NAO']

    for key, label in zip(new_dict_keys, labels):
        ax2[0].plot(mean_conc[key]['Diameter'], mean_conc[key]['number'], label = label)
        ax2[0].legend(fontsize = 8)
        ax2[0].set(xlabel = 'Particle diameter / $\mu$m', ylabel = 'dN/dlogDp, cm$^{-3}$', xscale='log', title = 'Particle number')
        ax2[1].plot(mean_conc[key]['Diameter'], mean_conc[key]['mass'], label = label)
        ax2[1].legend(fontsize = 8)
        ax2[1].set(xlabel = 'Particle diameter / $\mu$m', ylabel = 'dM/dlogDp, $\mu$g/m$^{3}$', xscale = 'log', title = 'Particle mass')

    fig2.tight_layout()

    return mean_conc, axes, figs

def plot_running_mean(fig, ax, df, bins, bin_edges, axis_labels, run_length, background):
    n_lines = len(df.keys())
    cmap = mpl.colormaps['plasma_r']
    colors = cmap(np.linspace(0, 1, n_lines))

    data = np.array(df[df.keys()]).T

    if bin_edges is not None:
        dlogDp = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        data = data / dlogDp

    if background:
        for i in range(len(df.keys()[1:])):
            ax.plot(bins, data[i+1], color = colors[i+1], lw = 1.2)

        ax2 = ax.twinx()
        ax2.plot(bins, data[0], color = 'k', alpha = 0.3, label = 'Background', lw = 1)

        ax2.tick_params(axis = 'y', labelsize = 8, labelcolor = 'dimgrey')
        ax2.set_ylabel('Background ' + axis_labels[1], color = 'dimgrey')

        # Create a scalar mappable for colorbar
        norm = mpl.colors.Normalize(vmin=run_length, vmax=run_length + (n_lines - 1) * run_length)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar

        # Create and place the colorbar
        cbar = fig.colorbar(sm, ax=ax2, orientation='vertical', pad=0.05)
        cbar.set_label('Time (min)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], xscale='log')
    else:
        for i in range(n_lines):
            ax.plot(bins, data[i], color=colors[i], lw=1.2)

        # Create a scalar mappable for colorbar
        norm = mpl.colors.Normalize(vmin=run_length, vmax=run_length + (n_lines - 1) * run_length)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar

        # Add colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Time (min)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], xscale='log')

def plot_reference(ax, x_plot, data, keys, labels):
    # Plot a scatter plot of the two concentrations
    ax.plot(x_plot, x_plot, color = 'grey', lw = 1, ls = '--')

    a, b, squares, ndof, R2 = linear_fit(data[keys[0]], data[keys[1]], 1, 0)
    y_fit = a*x_plot + b

    ax.plot(x_plot, y_fit, label = 'Fit', color = 'k', lw = 1.2)

    scatter_lbl = labels[0].split(' ')[0] + ' vs ' + labels[1].split(' ')[0]
    ax.scatter(data[keys[0]], data[keys[1]], s=10, c='blue', label = scatter_lbl) 

    # Set labels and title for the scatter plot
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', bottom = True, left = True)
    ax.set_xlabel(labels[0], fontsize=8)
    ax.set_ylabel(labels[1], fontsize=8)
    ax.set(xlim = (min(x_plot), max(x_plot)), ylim = (min(x_plot), max(x_plot)))

    ax.legend(fontsize = 8)

def plot_reference_same(ax, data_dict, dict_keys, concentration, timelabel, x_plot, axis_labels):

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

    plot_reference(ax, x_plot, merged, dict_keys, axis_labels)

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

    x_plot = np.linspace(0, max(merged_df[dict_keys[0]]), 100)
    
    plot_reference(ax, x_plot, merged_df, dict_keys, axis_labels)

def instrument_comparison(ax, data, data_keys, ref_data, concentration, timelabel, x_plot, axis_labels, timestamps):
    # Convert start_time and end_time to datetime objects if they are strings
    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    # Extract time and concentration data for both datasets
    time = pd.to_datetime(ref_data[timelabel[1]]).round('60s')
    conc = np.array(ref_data[concentration[1]])

    # Apply the time filter to both datasets
    time_filter = (time >= start_time) & (time <= end_time)
    filtered_time = time[time_filter]
    filtered_conc = conc[time_filter]

    # Create DataFrames to align both datasets by timestamp
    ref_df = pd.DataFrame({timelabel[0]: filtered_time, 'Reference': filtered_conc})

    for i, key in enumerate(data_keys):
        if 'dm' in key:
            new_df = running_mean(data, key, concentration[0], timelabel[0], '1T', 1) # '1T' is for 1-minute intervals

            merged = pd.merge(new_df, ref_df, on = timelabel[0], how = 'inner')

            plot_reference(ax[i], x_plot, merged, ['Reference', key], axis_labels)

        if 'ma200' in key:
            time = pd.to_datetime(data[key][timelabel[0]]).round('60s')
            conc = np.array(data[key][concentration[0]])
            new_df = pd.DataFrame({timelabel: time, key: conc})

            merged = pd.merge(new_df, ref_df, on = timelabel[0], how = 'inner')

            plot_reference(ax[i], x_plot, merged, ['Reference', key], axis_labels)
        
        if 'dm' not in key:
            if 'ma200' not in key:
                time = pd.to_datetime(data[key][timelabel[0]]).round('60s')
                conc = np.array(data[key][concentration[0]])

                # Apply the time filter to both datasets
                time_filter = (time >= start_time) & (time <= end_time)
                filtered_time = time[time_filter]
                filtered_conc = conc[time_filter]

                # Create DataFrames to align both datasets by timestamp
                new_df = pd.DataFrame({'timestamp': filtered_time, key: filtered_conc})
                
                merged = pd.merge(new_df, ref_df, on = timelabel[0], how = 'inner')

                plot_reference(ax[i], x_plot, merged, ['Reference', key], axis_labels)