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
def get_mean_conc(data, dict_keys, timelabel, timestamps, concentration, path):
    pd.options.mode.chained_assignment = None 
    
    idx_array = []
    for i, key in enumerate(dict_keys):
        idx_ts = np.zeros(len(timestamps[i]))
        for j, ts in enumerate(timestamps[i]):
            for k, time in enumerate(data[key][timelabel]):
                if ts in str(time):
                    idx_ts[j] += k
        idx_array.append(idx_ts)
    
        print(idx_ts)

    mean_df = pd.DataFrame()
    for i, key in enumerate(dict_keys):
        mean_conc = []
        time_start = []
        time_end = []
        for j, idx in enumerate(idx_array[i][::2]):
            new_df = data[key].iloc[int(idx):int(idx_array[i][j*2+1]), :] 
            time_start.append(data[key][timelabel][int(idx)]) 
            time_end.append(data[key][timelabel][int(idx_array[i][j*2+1])])
            mean = new_df[concentration].mean()
            mean_conc.append(mean)
        mean_df[key + ' time start'] = time_start
        mean_df[key + ' time end'] = time_end
        mean_df[key] = mean_conc

    mean_df.to_csv(path)

    return mean_df

def mean_conc_LCS(data, dict_keys, timelabel, date, timestamps, concentration, path):
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
    if squares_simple == 0:
        R2 = 'R2 not available'

            # Print the fitted parameters
        print(f"Fit: a={a_fit:6.6f}  b={b_fit:5.3f}  {R2}")

    if squares_simple != 0:
        R2 = 1 - (squares_fit / squares_simple)

        # Print the fitted parameters
        print(f"Fit: a={a_fit:6.6f}  b={b_fit:5.3f}  R2={R2:6.6f}")
    
    return a_fit, b_fit, squares_fit, Ndof_fit, R2

def running_mean(data, key, concentration, timelabel, interval, wndow):
    # Set 'Time' as the index
    new_df = pd.DataFrame() 
    new_df[timelabel] = pd.to_datetime(data[key][timelabel])
    new_df[key] = data[key][concentration]
    new_df = new_df.set_index(timelabel)

    # Resample the data to bins 
    new_df = new_df.resample(interval).mean() 
    
    # Now, apply the rolling mean
    new_df[key] = new_df[key].rolling(window = wndow, min_periods = 1).mean()

    return new_df

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

def bin_mean(timestamps, df, df_keys, timelabel):
    mean = []
    std = []
    error = []

    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    time = pd.to_datetime(df[timelabel])
    time_filter = (time >= start_time) & (time <= end_time)

    for key in df_keys:
        conc = np.array(df[key].dropna())
        
        # Convert the concentration data to numeric, coercing errors
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]
        bin_mean = filtered_conc.mean()
        bin_std = filtered_conc.std()
        error_mean = bin_std / np.sqrt(len(filtered_conc))
        mean.append(bin_mean)
        std.append(bin_std)
        error.append(error_mean)
    
    return mean, std, error

def plot_bin_mean(ax, timestamps, df, df_keys, timelabel, bins, axis_labels):
    mean, std, error = bin_mean(timestamps, df, df_keys, timelabel)

    ax.errorbar(bins, mean, error, fmt='.', ecolor='k', elinewidth=1, capsize=2, capthick=1)

    ax.set(xlabel = axis_labels[0], ylabel = axis_labels[1], xscale='log')

    # Set labels and title for the scatter plot
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', bottom = True, left = True)