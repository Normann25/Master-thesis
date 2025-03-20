import numpy as np
import pandas as pd
from iminuit import Minuit
#%%
def time_filtered_arrays(df, date, timestamps, conc_key):
    start_time = pd.to_datetime(date + ' ' + timestamps[0])
    end_time = pd.to_datetime(date + ' ' + timestamps[1])
    time = pd.to_datetime(df['Time'])

    time_filter = (time >= start_time) & (time <= end_time)
    filtered_time = np.array(time[time_filter])

    conc = np.array(df[conc_key])
    conc = pd.to_numeric(conc, errors='coerce')
    filtered_conc = conc[time_filter]
    return filtered_time, filtered_conc

def get_corrected_LCS(path, uncorrected_LCS, device_id, correction):
    corrected_LCS = {}

    for key in uncorrected_LCS.keys():
        if device_id in key:
            df = uncorrected_LCS[key]

            new_df = pd.DataFrame({'Time': df[df.keys()[0]]})

            for conc in df.keys()[1:]:
                new_df[conc] = np.array(df[conc])*correction

            new_df.to_csv(path + key + '.csv')

            corrected_LCS[key] = new_df
            
    return corrected_LCS

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

def linear_fit(x, y, a_guess, b_guess, forced_zero):

    Npoints = len(y)
    x, y = np.array(x), np.array(y)

    if forced_zero:
        def fit_func(x, a):
            return (a * x)
        
        def least_squares(a) :
            y_fit = fit_func(x, a)
            squares = np.sum((y - y_fit)**2)
            return squares
        
        least_squares.errordef = 1.0    # Chi2 definition (for Minuit)

        # Here we let Minuit know, what to minimise, how, and with what starting parameters:   
        minuit = Minuit(least_squares, a = a_guess)

        # Perform the actual fit:
        minuit.migrad()

        # Extract the fitting parameters:
        a_fit = minuit.values['a']

        Nvar = 1                     # Number of variables 
        Ndof_fit = Npoints - Nvar    # Number of degrees of freedom = Number of data points - Number of variables

        b_fit = 0
        
    else:
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
        minuit.migrad()

        # Extract the fitting parameters:
        a_fit = minuit.values['a']
        b_fit = minuit.values['b']

        Nvar = 2                     # Number of variables 
        Ndof_fit = Npoints - Nvar    # Number of degrees of freedom = Number of data points - Number of variables

    # Get the minimal value obtained for the quantity to be minimised (here the Chi2)
    squares_fit = minuit.fval                          # The chi2 value

    # Calculate R2
    R2 = ((Npoints * np.sum(x * y) - np.sum(x) * np.sum(y)) / (np.sqrt(Npoints * np.sum(x**2) - (np.sum(x))**2)*np.sqrt(Npoints * np.sum(y**2) - (np.sum(y))**2)))**2

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

def LCS_bins(data_dict, dict_keys, old_keys_flip, flip, old_keys_bin):
    new_dict = {}

    for i, key in enumerate(dict_keys):
        new_df = pd.DataFrame({'Time': data_dict[key]['Time']})
        if flip[i] == True:
            new_keys = ['PN0.5', 'PN1', 'PN2.5', 'PN5', 'PN10']
            for j in range(len(new_keys)):  
                new_df[new_keys[j]] = (data_dict[key][old_keys_flip[j]] - data_dict[key][old_keys_flip[j+1]])
            
            new_df['PN<1'] = new_df['PN0.5'] + new_df['PN1']
            new_df['PN<2.5'] = new_df['PN<1'] + new_df['PN2.5']
            new_df['PN<5'] = new_df['PN<2.5'] + new_df['PN5']
            new_df['PN<10'] = new_df['PN<5'] + new_df['PN10']

            new_dict[key] = new_df.dropna()
            
        else:    
            if 'DG-0001A' in key:
                new_dict[key] = data_dict[key]

            else:
                try:
                    new_keys = ['PN0.5', 'PN<1', 'PN<2.5', 'PN<5', 'PN<10']
                    for old, new in zip(old_keys_bin, new_keys):
                        new_df[new] = data_dict[key][old]
                
                except KeyError:
                    new_df['PN0.5'] = data_dict[key]['PN0.5, #/m3']
                    new_keys = ['PN<1', 'PN<2.5', 'PN<5', 'PN<10']
                    for old, new in zip(old_keys_bin[1:], new_keys):
                        new_df[new] = data_dict[key][old]

                new_dict[key] = new_df.dropna()

    return new_dict

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# Functions written by Anders Brostrøm
def Partector_TEM_sampling(Partector_data, ignore_samplings_below=0):
    """
    Function to identify active sampling periods of the PartectorTEM and return
    the sample duration as well as the start time for the given sample.

    Parameters
    ----------
    Partector_data : np.array
        An array with partector data as returned by the IL.Load_Partector function.
    ignore_samplings_below : int, optional
        When specified, the function ignores samplings with a duration shorter
        than the specified value in minutes. The default is 0.

    Returns
    -------
    valid_durations : np.array
        Array of the sampling durations within the dataset given in minutes.
    valid_starts : np.array
        Array of sample starting times from the dataset.
    """
    signal = Partector_data[:, 5].astype(int)  # Ensure the signal is integer
    timestamps = Partector_data[:, -1]  # Extract the timestamps (assumed to be datetime.datetime objects)

    # Find transitions
    start_indices = np.where((signal[:-1] == 0) & (signal[1:] == 1))[0] + 1
    end_indices = np.where((signal[:-1] == 1) & (signal[1:] == 0))[0] + 1
    
    # Step 3: Ensure proper pairing of starts and ends
    if len(start_indices) == 0 or len(end_indices) == 0:
        print("No transitions detected.")
        return  # Exit the function if no transitions are detected
    
    # Remove unmatched starts/ends if necessary
    if end_indices[0] < start_indices[0]:
        end_indices = end_indices[1:]  # Remove unmatched end at the start
    if start_indices[-1] > end_indices[-1]:
        start_indices = start_indices[:-1]  # Remove unmatched start at the end
    
    # Step 4: Calculate durations (in minutes) and filter based on ignore_samplings_below
    valid_durations = []
    valid_starts = []
    for start, end in zip(start_indices, end_indices):
        start_time = timestamps[start]
        end_time = timestamps[end]
        duration = (end_time - start_time).total_seconds() / 60  # Convert to minutes
        
        # Only keep durations longer than ignore_samplings_below
        if duration >= ignore_samplings_below:
            valid_durations.append(duration)
            valid_starts.append(start_time)
    
    # Step 5: Output results
    if not valid_durations:
        print("No valid sampling segments detected.")
    else:
        for j in range(len(valid_durations)):
            print("Start: {0}, Duration: {1:.2f} minutes".format(valid_starts[j],valid_durations[j]))
    return np.array(valid_durations), np.array(valid_starts)

def Partector_Ceff(Psize):
    """
    Function to estimate the collection efficiency of the partectorTEM at the
    specified particle size in nm. The collection efficiency as a fraction is 
    returned and can be applied to the measured concentration to get a 
    corrected concentration.
    
    It should be noted, that the expression for the collection efficiency was fitted
    to data from experiments with NaCl and Ag particles in the size range from 
    3 to 320 nm, and may therefore not be accurate at um sizes! Especially, at
    sizes larger than 4-5 um, the estimate will fail, as impaction will start to
    play a role. There are currently no data on the matter, but theoretical 
    caculations suggest that D50 is a roughly 11 um, but an effect can be seen
    already at 4-5 um.
   
    Reference: Fierz, M., Kaegi, R., and Burtscher, H.;"Theoretical and 
    Experimental Evaluation of a Portable Electrostatic TEM Sampler", Aerosol
    Science and Technology, 41, issue 5, 2007.

    Parameters
    ----------
    Psize : float or np.array
        Either a single particle size given as a float, or an array of particle
        sizes to be used for calculating the collection efficiency. The sizes should
        be given in nm.

    Returns
    -------
    Collection_efficiency : float or np.array
        The calculated collection efficiency of the PartectorTEM at the specified
        particle size/sizes specified as a fraction (0-1).

    """
    Collection_efficiency = (0.43837287*Psize**(-0.48585362))
    
    return Collection_efficiency 
