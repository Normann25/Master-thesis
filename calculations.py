import numpy as np
import pandas as pd
from scipy import stats
from iminuit import Minuit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from ExternalFunctions import *
#%%
# Fit functions
def linear_forced_zero(x, a):
    return (a * x)

def linear(x, a, b):
    return b + (a * x)

def gauss(x, p, mu, sigma):
    return p*stats.norm.pdf(x, mu, sigma)

def d_gauss(x, p1, mu1, sigma1, p2, mu2, sigma2):
    return p1*stats.norm.pdf(x, mu1, sigma1) + p2*stats.norm.pdf(x, mu2, sigma2)

def t_gauss(x, p1, mu1, sigma1, p2, mu2, sigma2, p3, mu3, sigma3):
    return p1*stats.norm.pdf(x, mu1, sigma1) + p2*stats.norm.pdf(x, mu2, sigma2) + p3*stats.norm.pdf(x, mu3, sigma3)

def lognorm(x, p, mu, sigma):
    return p*stats.lognorm.pdf(x, scale = mu, s = sigma)

def d_loggauss(x, p1, mu1, sigma1, p2, mu2, sigma2):
    return p1*stats.lognorm.pdf(x, scale = mu1, s = sigma1) + p2*stats.lognorm.pdf(x, scale = mu2, s = sigma2)

def t_loggauss(x, p1, mu1, sigma1, p2, mu2, sigma2, p3, mu3, sigma3):
    return p1*stats.lognorm.pdf(x, scale = mu1, s = sigma1) + p2*stats.lognorm.pdf(x, scale = mu2, s = sigma2) + p3*stats.lognorm.pdf(x, scale = mu3, s = sigma3)

def q_loggauss(x, p1, mu1, sigma1, p2, mu2, sigma2, p3, mu3, sigma3, p4, mu4, sigma4):
    return p1*stats.lognorm.pdf(x, scale = mu1, s = sigma1) + p2*stats.lognorm.pdf(x, scale = mu2, s = sigma2) + p3*stats.lognorm.pdf(x, scale = mu3, s = sigma3) + p4*stats.lognorm.pdf(x, scale = mu4, s = sigma4)

def lognorm_gauss(x, p1, mu1, sigma1, p2, mu2, sigma2):
    return p1*stats.lognorm.pdf(x, scale = mu1, s = sigma1) + p2*stats.norm.pdf(x, mu2, sigma2)

def dlognorm_gauss(x, p1, mu1, sigma1, p2, mu2, sigma2, p3, mu3, sigma3):
    return p1*stats.lognorm.pdf(x, scale = mu1, s = sigma1) + p2*stats.lognorm.pdf(x, scale = mu2, s = sigma2) + p3*stats.norm.pdf(x, mu3, sigma3)
#%%
def time_filtered_arrays(df, date, timestamps, conc_key):
    if date == None:
        start_time = pd.to_datetime(timestamps[0])
        end_time = pd.to_datetime(timestamps[1])  
    else:
        start_time = pd.to_datetime(f'{date} {timestamps[0]}')
        end_time = pd.to_datetime(f'{date} {timestamps[1]}')

    time = pd.to_datetime(df['Time'])

    time_filter = (time >= start_time) & (time <= end_time)
    filtered_time = np.array(time[time_filter])

    conc = np.array(df[conc_key])
    conc = pd.to_numeric(conc, errors='coerce')
    filtered_conc = conc[time_filter]
    return filtered_time, filtered_conc

def get_corrected(path, uncorrected, device_id, correction, data_type):
    corrected = {}

    for key in uncorrected.keys():
        if device_id in key:
            df = uncorrected[key]

            if data_type == 'LCS':

                new_df = pd.DataFrame({'Time': df[df.keys()[0]]})

                for conc in df.keys()[1:]:
                    new_df[conc] = np.array(df[conc])*correction

                new_df.to_csv(path + key + '.csv')

                corrected[key] = new_df

            if data_type == 'MA200':

                conc_keys= ['UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc', 'IR BCc']
                
                new_df = pd.DataFrame({
                    'Time': df['Time'], 
                    'Sample temp (C)': df['Sample temp (C)'],
                    'Sample RH (%)': df['Sample RH (%)'],
                    'Sample dewpoint (C)': df['Sample dewpoint (C)']})
                
                for i, conc in enumerate(conc_keys):
                    DP_correction = correction[0][i]*np.array(df['Sample dewpoint (C)']) + correction[1][i]

                    new_df[conc] = np.array(df[conc]) - DP_correction

                    new_df.to_csv(path + key + '.csv')

                    corrected[key] = new_df
            
    return corrected

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

def linear_fit(x, y, fitfunc, **kwargs):

    Npoints = len(y)
    x, y = np.array(x), np.array(y)
    
    def obt(*args):
        squares = np.sum(((y-fitfunc(x, *args)))**2)
        return squares

    minuit = Minuit(obt, **kwargs, name = [*kwargs]) # Setup; obtimization function, initial variable guesses, names of variables. 
    minuit.errordef = 1 # needed for likelihood fits. No explaination in the documentation.

    minuit.migrad() # Compute the fit
    valuesfit = np.array(minuit.values, dtype = np.float64) # Convert to numpy
    errorsfit = np.array(minuit.errors, dtype = np.float64) # Convert to numpy
    # if not minuit.valid: # Give custom error if the fit did not converge
    #     print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")

    Nvar = len(kwargs)           # Number of variables
    Ndof_fit = len(x) - Nvar

    squares_fit = minuit.fval  

    # Calculate R2
    R2 = ((Npoints * np.sum(x * y) - np.sum(x) * np.sum(y)) / (np.sqrt(Npoints * np.sum(x**2) - (np.sum(x))**2)*np.sqrt(Npoints * np.sum(y**2) - (np.sum(y))**2)))**2 

    return valuesfit, errorsfit, Ndof_fit, squares_fit, R2

def Chi2_fit(x, y, yerr, fitfunc, **kwargs):
    # Written by Philip Kofoed-Djursner
    
    def obt(*args):
        chi2 = np.sum(((y-fitfunc(x, *args))/yerr)**2)
        return chi2

    minuit = Minuit(obt, **kwargs, name = [*kwargs]) # Setup; obtimization function, initial variable guesses, names of variables. 
    minuit.errordef = 1 # needed for likelihood fits. No explaination in the documentation.

    minuit.migrad() # Compute the fit
    valuesfit = np.array(minuit.values, dtype = np.float64) # Convert to numpy
    errorsfit = np.array(minuit.errors, dtype = np.float64) # Convert to numpy

    Nvar = len(kwargs)           # Number of variables
    Ndof_fit = len(x) - Nvar

    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit) 

    return valuesfit, errorsfit, Prob_fit

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

def AAE_calc(df, timestamps):
    AAE_df = pd.DataFrame()

    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1]) 

    time = pd.to_datetime(df['Time'])
    time_filter = (time >= start_time) & (time <= end_time)

    time_filtered_df = df[time_filter]

    conc_filter = time_filtered_df['IR BCc'] >= 0.25
    filtered_df = time_filtered_df[conc_filter]

    # Wavelengths
    wvl = [375, 470, 528, 625, 880]

    # Specific attenuation cross-section
    sigma = np.array([24.069, 19.070, 17.028, 14.091, 10.120]) # m**2/g
    Cref = 1.3 # Multiple scattering coefficient

    # Mass absorbtion cross section (MAC)
    MAC = sigma/Cref # m**2/g
    
    # Absorption coefficients for UV and IR
    conc_keys = ['UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc']
    abs_880 = np.array(filtered_df['IR BCc'])*10**(-6)*MAC[-1] # m**-1, IR

    for i, key in enumerate(conc_keys):
        b_abs = np.array(filtered_df[key])*10**(-6)*MAC[i]

        # Absorption Ångstrøm exponent (AAE)
        AAE = -(np.log(b_abs/abs_880)/np.log(wvl[i]/880))

        conc = key.split(' ')[0]
        df_key = f'{conc} and IR'
        AAE_df[df_key] = AAE

    return AAE_df

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

def K_means_optimal(Kmeans_data,max_cluster=20):
    fig,axs = plt.subplots(ncols=3, figsize = (7, 4))
    
    # Compute WCSS for different values of k
    wcss = []
    for k in range(1, max_cluster):  # Test k values from 1 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(Kmeans_data)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Curve
    axs[0].plot(range(1, max_cluster), wcss, marker='o')
    axs[0].set_title('Elbow Method \n Find point where plot bends')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    
    silhouette_scores = []
    for k in range(2, max_cluster):  # Test k values from 2 to cluster (Silhouette requires at least 2 clusters)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(Kmeans_data)
        score = silhouette_score(Kmeans_data, labels)
        silhouette_scores.append(score)
    
    # Plot the Silhouette Scores
    axs[1].plot(range(2, max_cluster), silhouette_scores, marker='o')
    axs[1].set_title('Silhouette Score Method, \n Close to 1 is good')
    axs[1].set_xlabel('Number of Clusters (k)')
    axs[1].set_ylabel('Silhouette Score')
    
    db_scores = []
    for k in range(2, max_cluster):  # Test k values from 2 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(Kmeans_data)
        score = davies_bouldin_score(Kmeans_data, labels)
        db_scores.append(score)
    
    # Plot the Davies-Bouldin Scores
    axs[2].plot(range(2, max_cluster), db_scores, marker='o')
    axs[2].set_title('Davies-Bouldin Index Method \n Small values are good')
    axs[2].set_xlabel('Number of Clusters (k)')
    axs[2].set_ylabel('Davies-Bouldin Score')

    fig.tight_layout()
    plt.show()

    return fig
