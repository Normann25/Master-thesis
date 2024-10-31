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


def read_CPC(path, parent_path, timelable):
   parentPath = os.path.abspath(parent_path)
   if parentPath not in sys.path:
      sys.path.insert(0, parentPath)
   
   files = os.listdir(path)
   new_dict = {}



   #%%
   parent = '../../../' 

      
        
        