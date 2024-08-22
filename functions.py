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