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

def running_mean(df, dictkey, concentration, timelabel, interval, wndw, timestamps):
    # Set 'Time' as the index
    new_df = pd.DataFrame() 

    if timestamps == None:
        new_df[timelabel] = pd.to_datetime(df[timelabel])
        new_df[dictkey] = df[concentration]
        new_df = new_df.set_index(timelabel)

        # Resample the data to bins 
        new_df = new_df.resample(interval).mean() 
        
        # Now, apply the rolling mean
        new_df[dictkey] = new_df[dictkey].rolling(window = wndw, min_periods = 1).mean()

    if timestamps != None:
        start_time = pd.to_datetime(timestamps[0])
        end_time = pd.to_datetime(timestamps[1])

        time = pd.to_datetime(df[timelabel])

        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = pd.to_datetime(np.array(time[time_filter]))

        new_df = pd.DataFrame({'Timestamp': filtered_time})
        
        # Dictionary to collect new columns
        new_columns = {}

        for key in concentration:
            conc = np.array(df[key])
            conc = pd.to_numeric(conc, errors='coerce')
            filtered_conc = conc[time_filter]

            # Store the filtered data in the dictionary
            new_columns[key] = filtered_conc

        # Convert dictionary to DataFrame and concatenate it with `new_df`
        new_df = pd.concat([new_df, pd.DataFrame(new_columns)], axis=1)
        new_df = new_df.set_index('Timestamp')

        # Resample the data to bins 
        new_df = new_df.resample(interval).mean() 
        
        for key in concentration:
            # Now, apply the rolling mean
            new_df[key] = new_df[key].rolling(window = wndw, min_periods = 1).mean()

    return new_df

def bin_mean(timestamps, df, df_keys, timelabel, inst_error):
    mean = np.zeros(len(df_keys))
    std = np.zeros(len(df_keys))

    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])
    if timelabel != None:
        time = pd.to_datetime(df[timelabel])
    else:
        time = pd.to_datetime(df.index)
    time_filter = (time >= start_time) & (time <= end_time)

    for i, key in enumerate(df_keys):
        conc = np.array(df[key])
        
        # Convert the concentration data to numeric, coercing errors
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]
        mean[i] += filtered_conc.mean()
        std[i] += filtered_conc.std() / np.sqrt(len(filtered_conc))
    
    if inst_error != None:
        errors = mean * inst_error
    else:
        errors = 0

    return mean, std, errors

def calc_mass_conc(df, df_keys, bin_mid_points, rho):
    try:
        new_df = pd.DataFrame({'Time': df['Time']})
    except KeyError:
        new_df = pd.DataFrame({'Time': df.index})

    new_columns = {}    
    for i, key in enumerate(df_keys):
        # Ensure df[key] is numeric
        df[key] = np.array(pd.to_numeric(df[key], errors='coerce'))
        
        new_columns[key] = (rho / 10**6) * (np.pi / 6) * bin_mid_points[i]**3 * df[key] * 10**6 # in ug * m**-3
    # Convert dictionary to DataFrame and concatenate it with `new_df`
    new_df = pd.concat([new_df, pd.DataFrame(new_columns)], axis=1)

    return new_df

def bin_edges(d_min, bin_mid):
    bins_list = [d_min]

    for i, bin in enumerate(bin_mid):
        bin_max = bin**2 / bins_list[i] 
        bins_list.append(bin_max)
    
    return bins_list

def binned_mean(timestamps, dict_number, dict_mass, dict_keys, bins, start_point, cut_point, timelabel, mass):
    running_number = {}
    running_mass = {}
    for i, key in enumerate(dict_keys):
        new_key = 'Exp' + str(i + 1)
        df_number = pd.DataFrame({'Time': dict_number[key]['Time']})
        for size, cut in zip(bins, cut_point):
            df_number[size] = dict_number[key].iloc[:,start_point[0]:cut].sum(axis = 1)
        mean_number, std, errors = bin_mean(timestamps[0][i], df_number, bins, timelabel, None)
        increase_number, std, errors = bin_mean(timestamps[1][i], df_number, bins, timelabel, None)
        bg_number = pd.DataFrame({'Background': mean_number, 'Increase': increase_number}).T
        bg_number.columns = bins
        exp_number = running_mean(df_number, None, bins, 'Time', '10T', 10, timestamps[2][i])
        running_number[new_key] = pd.concat([bg_number, exp_number])

        if mass == True:
            df_mass = pd.DataFrame({'Time': dict_mass[key]['Time']})
            for size, cut in zip(bins, cut_point):
                df_mass[size] = dict_mass[key].iloc[:,start_point[1]:cut].sum(axis = 1)
            mean_mass, std, errors = bin_mean(timestamps[0][i], df_mass, bins, timelabel, None)
            increase_mass, std, errors = bin_mean(timestamps[1][i], df_mass, bins, timelabel, None)
            bg_mass = pd.DataFrame({'Background': mean_mass, 'Increase': increase_mass}).T
            bg_mass.columns = bins
            exp_mass = running_mean(df_mass, None, bins, 'Time', '10T', 10, timestamps[2][i])
            running_mass[new_key] = pd.concat([bg_mass, exp_mass])

    return running_number, running_mass

def split_data_timestamps(df, timestamps, timelabel, concentration):
    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    time = pd.to_datetime(df[timelabel])

    time_filter = (time >= start_time) & (time <= end_time)

    filtered_time = pd.to_datetime(np.array(time[time_filter]))

    new_df = pd.DataFrame({'Time': filtered_time})
    # new_df = new_df.set_index('Time')
    
    for key in concentration:
        conc = np.array(df[key])
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]

        new_df[key] = filtered_conc

    return new_df

def merge_data(dict_small_Dp, small_Dp_keys, small_Dp_interval, dict_large_Dp, large_Dp_keys, large_Dp_interval, timestamps, timelabel, running, round):
    new_dict_number = {}
    new_dict_mass = {}

    for i, key in enumerate(small_Dp_keys):
        small_df_keys = dict_small_Dp[key].keys()[small_Dp_interval[0]:small_Dp_interval[1]].to_list()
        large_df_keys = dict_large_Dp[large_Dp_keys[i]].keys()[large_Dp_interval[0]:large_Dp_interval[1]].to_list()
        name = 'Exp' + str(i+1)

        if round == False:
            df_small = running_mean(dict_small_Dp[key], None, small_df_keys, timelabel[0], '1T', 1, timestamps[i])
            df_small = df_small.fillna(0)
            df_small['Time'] = pd.to_datetime(df_small.index)
            df_large = running_mean(dict_large_Dp[large_Dp_keys[i]], None, large_df_keys, timelabel[1], '1T', 1, timestamps[i])
            df_large = df_large.fillna(0)
            df_large['Time'] = pd.to_datetime(df_large.index)

        if running == False:
            dict_small_Dp[key][timelabel[0]] = pd.to_datetime(dict_small_Dp[key][timelabel[0]]).round('60s')
            dict_large_Dp[large_Dp_keys[i]][timelabel[1]] = pd.to_datetime(dict_large_Dp[large_Dp_keys[i]][timelabel[1]]).round('60s')

            df_small = split_data_timestamps(dict_small_Dp[key], timestamps[i], timelabel[0], small_df_keys)
            df_large = split_data_timestamps(dict_large_Dp[large_Dp_keys[i]], timestamps[i], timelabel[1], large_df_keys)

        if running and round:
            df_small = running_mean(dict_small_Dp[key], None, small_df_keys, timelabel[0], '1T', 1, timestamps[i])
            df_small = df_small.fillna(0)
            df_small['Time'] = pd.to_datetime(df_small.index)

            dict_large_Dp[large_Dp_keys[i]][timelabel[1]] = pd.to_datetime(dict_large_Dp[large_Dp_keys[i]][timelabel[1]]).round('60s')
            df_large = split_data_timestamps(dict_large_Dp[large_Dp_keys[i]], timestamps[i], timelabel[1], large_df_keys)

        small_df_floats = []
        small_df_strings = []
        for key in small_df_keys:
            new_key = float(key) / 1000
            small_df_floats.append(new_key)
            small_df_strings.append(str(new_key))
            df_small = df_small.rename(columns = {key: str(new_key)})
        
        large_df_floats = []
        for key in large_df_keys:
            large_df_floats.append(float(key))

        merged_keys = small_df_strings + large_df_keys
        merged_bin_mean = small_df_floats + large_df_floats

        merged = pd.merge(df_small, df_large, on = 'Time')
        merged = merged.reset_index()

        merged_mass = calc_mass_conc(merged, merged_keys, merged_bin_mean, 1.2)

        new_dict_number[name] = merged
        new_dict_mass[name] = merged_mass

    return new_dict_number, new_dict_mass, merged_keys, merged_bin_mean