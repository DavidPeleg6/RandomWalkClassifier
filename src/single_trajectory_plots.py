# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:02:29 2021

@author: gedadav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def setup_env():
    # Use seaborn style defaults and set the default figure size
    sns.set()
    sns.set_context('paper', font_scale=2.5)
    # increase plot size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    
    # create the folder to save results
    location = 'results/' + datetime.now().strftime("%d_%m")
    full_data_loc = location + '/single_trajectory_plots/'
    try:
        os.mkdir('results')
    except OSError:
        pass
    try:
        os.mkdir(location)
    except OSError:
        pass
    try:
        os.mkdir(full_data_loc)
    except OSError:
        pass
    return full_data_loc

# setup the all the locations and folders for the results to be placed
plot_loc = setup_env()

df = pd.read_csv('test_set.csv')
plt.figure()
subframe = df.iloc[10:20, :]
for walk in subframe.iloc[:, 1:-5].to_numpy():
    # if np.amax(np.abs(walk)) < 5:
        print(subframe.iloc[:, -4])
        plt.plot([i for i in range(len(walk))], walk)
            
plt.xlabel('t')
plt.ylabel('x')
# plt.title(label)
plt.savefig(plot_loc + 'all_walks.png')
plt.show()
            


