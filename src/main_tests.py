# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:04:46 2020

@author: gedadav
"""

import DatasetCreator
from Generator import UniformGenerator
from LivnatNetClassifier import main_loop
import matplotlib.pyplot as plt
from datetime import datetime
import os
import seaborn as sns
import numpy as np

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
    dataset_plot_loc = location + '/dataset_plots/'
    loss_plot_loc = location + '/loss plots/'
    full_data_loc = location + '/full results/'
    try:
        os.mkdir('results')
    except OSError:
        pass
    try:
        os.mkdir(location)
    except OSError:
        pass
    try:
        os.mkdir(dataset_plot_loc)
    except OSError:
        pass
    try:
        os.mkdir(loss_plot_loc)
    except OSError:
        pass
    try:
        os.mkdir(full_data_loc)
    except OSError:
        pass
    return dataset_plot_loc, loss_plot_loc, full_data_loc



""" first dataset configuration
{'Normal noise constant interval': 'Brownian',
         'symmetric exponential noise constant interval': 'Brownian',
         'exponential noise constant interval': 'Brownian',
         'levy noise constant interval': 'Levy',
         'FBM negative correlation constant interval': 'negative FBM',
         'FBM positive correlation constant interval': 'positive FBM',
         'FBM variable hurst exponent': 'FBM',
         'CTRW normal noise constant alpha': 'CTRW',
         'CTRW normal noise variable alpha': 'CTRW',
         'CTRW normal noise uniform waiting times': 'CTRW',
         'CTRW symmetric exponential uniform waiting times': 'CTRW',
         'CTRW exponential noise uniform waiting times': 'CTRW',
         'CTRW levy noise uniform waiting times': 'CTRW' }
"""

""" second dataset configuration
{'Normal noise constant interval': 'Brownian'
         ,'symmetric exponential noise constant interval': 'Brownian'
         ,'fat tail powerlaw noise constant interval': 'Levy'
         ,'FBM variable hurst exponent': 'FBM'
         ,'CTRW normal noise variable alpha': 'CTRW'
         ,'CTRW symmetric noise variable alpha': 'CTRW'
         ,'CTRW levy noise variable alpha': 'CTRW' 
         }
"""


# create the a dataset to run the network on
names = {'Normal noise constant interval': 'Brownian',
         # 'symmetric exponential noise constant interval': 'Brownian',
         # 'exponential noise constant interval': 'Brownian',
         'levy noise constant interval': 'Levy',
         # 'FBM negative correlation constant interval': 'negative FBM',
         # 'FBM positive correlation constant interval': 'positive FBM',
         'FBM variable hurst exponent': 'FBM',
         # 'CTRW normal noise constant alpha': 'CTRW',
         # 'CTRW normal noise variable alpha': 'CTRW',
         # 'CTRW normal noise uniform waiting times': 'CTRW',
         # 'CTRW symmetric exponential uniform waiting times': 'CTRW',
         # 'CTRW exponential noise uniform waiting times': 'CTRW',
         'CTRW levy noise uniform waiting times': 'CTRW' }

# setup the all the locations and folders for the results to be placed
dataset_plot_loc, loss_plot_loc, full_data_loc = setup_env()
#-----------------------------------HYPER PARAMETERS FOR DATASET--------------------------
# whether or not to recreate the dataset
recreate_dataset = True
# recreate_dataset = False
# whether or not to enable transitions in the random walk generating model
mixed_models = True
# mixed_models = False
# whether or not to enable variable walk lengths
# variable_lengths = True
variable_lengths = False
# to use the regression version of the models (for finding exact tranisiton point)
#regression = True
regression = False
# choose to classify based on model as label, or with true false label (for transition recognition)
classify_models = True
# classify_models = False
classify_by = 'model' if classify_models else 'mixed'
variable_diffusion = True
# variable_diffusion = False
# the preprocessing functions that will be applied to the dataset
# 'normalize_walks' , 'TA_MSD' , 'auto_corr_walks'
preprocess = ['auto_corr_walks']
repeat_experiment = 30
transitions = 2
# # for a dataset with only transitions - transition_prob = 1
transition_prob = 0.5 if mixed_models else 0
diffusion = UniformGenerator(0, 1) if variable_diffusion else 0.3
delta_t = 0.05  
walkers = int(7e4)
train, test = [0.75, 0.25]
steps = 100
min_steps = int(0.3 * steps) if variable_lengths else steps
start_points = [-0, 0]
processes = names
# turn back to True if youre using a regression network and trying to get the diffusion rate based on the path
classify_diffusion = False
train_name = f'train_set_{walkers}_{steps}.csv'
test_name = f'test_set_{walkers}_{steps}.csv'
param_name = f'parameters_{walkers}_{steps}.json'

hurst_neg = 0.4
hurst_pos = 0.6
variable_hurst = 'FBM variable hurst exponent' in names.keys()
variable_alpha = 'CTRW normal noise variable alpha' in names.keys()
# uni_time_gen = UniformGenerator(0,1)
# the alpha of the waiting times power law distribution
alpha = (1.1, 3)
levy_alpha = [2, 3]
# alpha = (0.5, 3)
# levy_alpha = [1, 3]

# ----------------------------------HYPER PARAMETERS FOR NETWORKS----------------
EPOCHS = 15
alpha_0 = 5e-3
# TODO change this when you figure out how to pack sequences so the GRU supports variable walk lengths in batches
# batch_size = 64 if not variable_length else 1
batch_size = 128
# 'SGD' , 'Adam' , 'RMSprop', 'ASGD', 'Adamax', 'AdamW', 'Adagrad'
optimizers = ['Adam']
# 'N_layer_GRU' , 'Convolutional_NN' , 'Bi_Directional_GRU' , 'Fully_Connected_NN'
# 'vanilla_RNN' , 'N_layer_RNN' , 'N_layer_LSTM' , 'Single_step_GRU', 'Bi_Single_step_GRU'
# 'Special_N_layer_GRU', 'Special_N_layer_GRU2', 'Special_N_layer_GRU3', 'Transition_GRU'
# 'Regression_Convolution' , 'Regression_GRU' , 'Regression_LSTM' , 'Regression_Fully_Connected'
# 'Wide_Fully_Connected', 'Wide_Convolution', 'ConvolutionalGRU'
net_types = ['ConvolutionalGRU', 'N_layer_GRU']
# RNN specific parameters
hidden_size = 40
layers = 3

# -----------------------------CREATE AND CHECK DATASETS----------------------
if recreate_dataset:
    print(f'recreating dataset: {train_name}')
    DatasetCreator.create_variable_length_dataset(diffusion=diffusion, delta_t=delta_t, steps=steps,
                                  train_walkers=int(walkers*train), test_walkers=int(walkers*test),
                                  names=processes, train_name=train_name, test_name=test_name,
                                  max_transitions=2, transition_prob=transition_prob, param_name=param_name,
                                  alpha=alpha, min_steps=min_steps, levy_alpha=levy_alpha)
    # DatasetCreator.create_variable_length_dataset(diffusion=diffusion, delta_t=delta_t, steps=steps,
    #                               train_walkers=int(walkers*train), test_walkers=int(walkers*test),
    #                               names=processes, train_name=train_name, test_name=test_name,
    #                               max_transitions=2, transition_prob=transition_prob, param_name=param_name,
    #                               alpha=alpha, min_steps=min_steps, start_point=UniformGenerator(start_points[0], start_points[1]))
    DatasetCreator.plot_dataset(train_name, dataset_plot_loc)

#------------------------------------RUN THE TESTS-----------------------------
# for further testing, keep the networks in the memory
nets, plotters, loader = [], [], None
print("walk amount: {} walk length: {} alpha range: {} ".format(int(walkers), steps, alpha))
# change the labels to be numerical so the network can easily feed on it
#label_dict = labels_to_nums(names)
accuracies = {net: [] for net in net_types}
for i in range(repeat_experiment):
    for net_name in net_types:
        print('experiment number: {} network type: {}'.format(i+1, net_name))
        for optimizer in optimizers:
            # set up a dictionary containing all the current experiment data to be saved
            exper_data = {'epochs': EPOCHS, 'batch size': batch_size, 'learning rate': alpha_0, 'optimizer': optimizer,
                          'net type': net_name, 'variable hurst exponent': variable_hurst, 'variable waiting times alpha':  variable_alpha,
                          'alpha range': alpha, 'levy alpha': levy_alpha, 'processes': names, 'walkers length': steps, 'walker amount': int(walkers), 
                          'sample interval': delta_t, 'model transition enabled': mixed_models, 'variable diffusion': variable_diffusion,
                          'start time': datetime.now().strftime(' %H_%M'), 'current epoch': 0, 'runtime': 0, 'pre-process': preprocess,
                          'train size': train, 'test size': test, 'hidden size': hidden_size, 'layers': layers, 'classify by': classify_by,
                          'variable length': variable_lengths, 'transition probability': transition_prob, 'min walk length': min_steps,
                          'start points': start_points
                          }
            net, plotter, test_loader, acc = main_loop(param_file=param_name, train_datafile=train_name, test_datafile=test_name,
                                                  location=loss_plot_loc,
                                                  layers=layers, hidden_size=hidden_size, regression=regression, exper_data=exper_data,
                                                  full_data_location=full_data_loc)
            accuracies[net_name].append(acc)
            nets.append(net)
            plotters.append(plotter)
            loader = test_loader
    plt.close('all')


np_mean = {net_name: np.mean(accuracies[net_name]) for net_name in net_types}
np_variance = {net_name: np.std(accuracies[net_name]) for net_name in net_types}
print(f'experiment repetitions: {repeat_experiment}, steps: {steps}, walker amount: {walkers}\n'
      f'mean: {np_mean}, \nvariance: {np_variance}')

