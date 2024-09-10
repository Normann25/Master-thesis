import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from matplotlib.ticker import FuncFormatter
import time
from datetime import datetime
#%%
def read_txt(path, parent_path, file_names, separation):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for name in file_names:
        for file in files:
            if name in file:
                with open(os.path.join(path, file)) as f:
                    df = pd.read_table(f, sep = separation)
                    new_dict[name] = df
    
    return new_dict

def read_txt_acsm(path, parent_path, file_names, separation):
    new_dict = {}
    data_dict = read_txt(path, parent_path, file_names, separation)

    for key in data_dict.keys():
        df = data_dict[key]
        df.columns = ['Time', 'org_conc']
        df['Time'] = df['Time'].str.split().str[1] + pd.Timedelta('2 hours')
        df['Time'] = pd.to_timedelta(df['Time']).astype('timedelta64[s]')
        new_dict[key] = df
    
    return new_dict

def read_data(path, parent_path, time_label):
    parentPath = os.path.abspath(parent_path)
    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)

    files = os.listdir(path)
    data_dict = {}

    for file in files:
        name = file.split('.')[0]
        with open(os.path.join(path, file)) as f:
            df = pd.read_csv(f, sep = ';', decimal=',')
            df = df.dropna()

            df['PAH total'] = pd.to_numeric(df['PAH total'], errors = 'coerce')

            df[time_label] = df[time_label].str.split().str[1] + pd.Timedelta('2 hours')
            df['Time'] = pd.to_timedelta(df[time_label]).astype('timedelta64[s]')

        data_dict[name] = df
    
    return data_dict

def plot_PAH_ACSM(ax, data_dict, dict_keys):
    for i, dict_key in enumerate(dict_keys):
        df = data_dict[dict_key]
        #for j, key in enumerate(df.keys()[1:]):
        ax[i].plot(df['Time'], df['PAH total'], lw = 1)


        formatter = FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
        ax[i].xaxis.set_major_formatter(formatter)
        ax[i].tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)

        ax[i].set_xlabel('Time', fontsize = 8)
        ax[i].set_ylabel('PAH$_{est}$ / $\mu$g/m$^{3}$', fontsize = 8)
        ax[i].set_title(dict_key, fontsize = 9)
