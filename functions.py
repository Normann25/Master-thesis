import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#%%
def read_txt(path, parent_path, file_names):
    new_dict = {}

    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    for name in file_names:
        for file in files:
            if name in file:
                with open(os.path.join(path, file)) as f:
                df = pd.read_table(f, sep = '\t')
                new_dict[name] = df
    
    return new_dict
