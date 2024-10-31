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




      
