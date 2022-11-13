# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:56:51 2021

@author: gedadav
"""

from LivnatNetClassifier import ConvNet, NLayerGRU, BiGRU
import matplotlib.pyplot as plt
import RandomWalks
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop, ASGD, Adamax, AdamW, Adagrad, LBFGS
import DatasetCreator
import torch.nn.functional as F
from CustomDataLoaders import MyDataset, MixedModelDataset
from torch.utils.data import DataLoader
from Generator import NormalGenerator, UniformGenerator
from RandomWalks import *
import time
from Plotter import TwoValPlotter
import os
from datetime import datetime
import sys
import json
import seaborn as sns
import CustomDataLoaders
import pandas as pd
import trainer
import itertools


def get_loaders(train_datafile, test_datafile, full_params, batch_size):
    label_dict = full_params['label dictionary']
    train_data = MixedModelDataset(train_datafile, label_dict)    
    test_data = MixedModelDataset(test_datafile, label_dict)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, label_dict


def load_data(label_name, param_name, network_file, net_type):
    with open(label_name, 'r') as f:
        labels = json.load(f)
    # label_dict = labels_to_nums(labels)
    with open(param_name, 'r') as res:
        full_params = json.load(res)
    # create a dictionary containing all the parameters of the experiment so it can be reproduced
    full_params.update(labels)
    
    if net_type is NLayerGRU:
        model = NLayerGRU(1, full_params['hidden size'], full_params['names'], num_layers=full_params['layers'])
    elif net_type is ConvNet:
        model = ConvNet(full_params['walkers length'], full_params['names'], full_params['batch size'])
    elif net_type is BiGRU:
        model = BiGRU(1, full_params['hidden size'], full_params['names'], num_layers=full_params['layers'])
    
    # then load the weights from the correct file
    saved_model = torch.load(network_file)
    # then set the network's weights accordingly
    model.load_state_dict(saved_model)
    return model, full_params


def create_confusion_matrix(validation_loader, model, label_dict, percent=False, custom_validation=True, splits=4):
    """
    calculates the confusion matrix
    
    Parameters
    ----------
    validation_loader : DataLoader
        a data loader for the model
    model : torch.nn
        a trained classifier
    label_dict : dict
        a python dictionary mapping from classes to numbers
    percent : bool, optional
        decides whether the output should be presented in percentage form. The default is False.

    Returns
    -------
    confusion : DataFrame
        a dataframe containing the confusion matrix where:
            rows = model prediction
            columns = actual classification
    accuracy : float
        the total accuracy of the model on the dataset

    """
    # initialize the confusion matrix in the shape of the labels
    labels = list(itertools.permutations(list(label_dict.keys()), 2))
    labels.extend([(clas, clas) for clas in label_dict.keys()])
    correct = 0
    confusion = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    new_dict = nums_to_labels(label_dict)
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            X, y = data
            output = trainer.validate_mixed(X, model, splits)
            pred, actual = (new_dict[int(output[0])], new_dict[int(output[1])]), (new_dict[int(y[0][0])], new_dict[int(y[0][1])])
            if output[0] == y[0][0] and output[1] == y[0][1]:
                correct += 1
            confusion[pred][actual] += 1
    # if we need the matrix annotated in percents rather than samples
    if percent:
        confusion = (100 * confusion) / len(validation_loader.dataset)
    accuracy = 100 * (correct / len(validation_loader.dataset))
    return confusion, accuracy    


def plot_confusion(confusion_mat, net_name, file_name):
    # Use seaborn style defaults and set the default figure size
    sns.set()
    sns.set_context('paper', font_scale=2.5)
    # # increase plot size
    fig_size = plt.rcParams["figure.figsize"]
    # size of x dimension of the matrix
    fig_size[0] = 14
    # size of y dimension of the matrix
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    # plot the heatmap with annotations (showing the actual values) without decimal points
    sns.heatmap(confusion_mat, annot=True, fmt=".0f")
    # plt.title('rows=actual, columns=prediction model={}'.format(net_name))
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def labels_to_nums(labels):
    return {str(labels[i]): i for i in range(len(labels))}

def nums_to_labels(labels):
    return {num:label for label,num in labels.items()}


def setup_env(location, net_name):
    # Use seaborn style defaults and set the default figure size
    sns.set()
    sns.set_context('paper', font_scale=2.5)
    # # increase plot size
    # fig_size = plt.rcParams["figure.figsize"]
    # # size of x dimension of the matrix
    # fig_size[0] = 14
    # # size of y dimension of the matrix
    # fig_size[1] = 10
    # plt.rcParams["figure.figsize"] = fig_size
        # # increase plot size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    
    # create the folder to save results
    full_location = location + '/' + net_name
    network_location = full_location + '/network/'
    confusion_mat_loc = full_location + '/confusion matrix/'
    dataset_loc = full_location + '/dataset/'
    param_loc = full_location + '/parameters/'
    try:
        os.mkdir(location)
    except OSError:
        pass
    try:
        os.mkdir(full_location)
    except OSError:
        pass
    try:
        os.mkdir(confusion_mat_loc)
    except OSError:
        pass
    try:
        os.mkdir(dataset_loc)
    except OSError:
        pass
    try:
        os.mkdir(network_location)
    except OSError:
        pass
    try:
        os.mkdir(param_loc)
    except OSError:
        pass
    return network_location, confusion_mat_loc, dataset_loc, param_loc


    
#   # create the a dataset to run the network on
# names = {
#         # 'Normal noise constant interval': 'Brownian'
#           # ,
#           # 'symmetric exponential noise constant interval': 'Brownian'
#           # ,
#           # 'fat tail powerlaw noise constant interval': 'Levy'
#           # ,
#             # 'FBM negative correlation constant interval': 'FBM'
#           # ,
#            # 'FBM positive correlation constant interval': 'FBM'
#           # ,
#            # 'CTRW normal noise constant alpha': 'CTRW'
#           }


# setup the all the locations and folders for the results to be placed
#----------------GRU network tests
# net_name = 'N_Layer_GRU_mixed'
# net_name = 'N_Layer_GRU'
# net_type = NLayerGRU
#----------------BiGRU network tests
# net_name = 'BiGRU_mixed'
net_name = 'BiGRU_non_mixed'
net_type = BiGRU
#----------------Convolution network tests
# net_name = 'Convolution_non_mixed'
# net_name = 'Convolution_mixed'
# net_type = ConvNet
location = 'mixed models test'
custom_validation = False
splits = 4
network_location, confusion_mat_loc, dataset_loc, param_loc = setup_env(location, net_name)   
recreate_dataset = True
# recreate_dataset = False
  
# you first initialize the network you wish to use
network_location = network_location + 'net.pt'
train_name = dataset_loc + 'train_set.csv'
test_name = dataset_loc + 'test_set.csv'
label_name = param_loc + 'parameters.json'
param_name = param_loc + 'results.json'
matrix_name = confusion_mat_loc + 'confusion.png'
dataset_plot_loc = dataset_loc
# load all the parameters and the model from the files
model, full_params = load_data(label_name, param_name, network_location, net_type)
# change the label_name location so there will not be overwrite of the param file
label_name = param_loc + 'new_params.json'
# check whether you classify the models that took part in the process or whether or not there has been a transition
classify_models = False if type(full_params['names'][0]) is bool else True
diffusion = UniformGenerator(0,1)
# diffusion = 0.3
delta_t = full_params['delta_t']
transition_prob = 0.5
max_transitions = 1
transition_range = [1/6, 5/6]
walkers = int(1e3)
steps = 120
min_steps = 120
train, test = [0.9, 0.1]
names = full_params['processes']
classify_diffusion = False
# the alpha of the waiting times power law distribution
alpha = full_params['alpha range']
levy_alpha = full_params["levy alpha"]

batch_size = 1

if recreate_dataset:
    DatasetCreator.create_variable_length_dataset(diffusion=diffusion, delta_t=delta_t, steps=steps,
                                  train_walkers=int(walkers*train), test_walkers=int(walkers*test),
                                  names=names, train_name=train_name, test_name=test_name,
                                  max_transitions=max_transitions, transition_prob=transition_prob, param_name=label_name,
                                  alpha=alpha, min_steps=min_steps, levy_alpha=levy_alpha, tran_range=transition_range)
    # DatasetCreator.create_variable_length_dataset(diffusion=diffusion, delta_t=delta_t, steps=steps,
    #                               train_walkers=int(walkers*train), test_walkers=int(walkers*test),
    #                               names=names, train_name=train_name, test_name=test_name,
    #                               max_transitions=2, transition_prob=transition_prob, param_name=label_name,
    #                               alpha=alpha, min_steps=min_steps, start_point=UniformGenerator(-2,2))
    DatasetCreator.plot_dataset(train_name, dataset_plot_loc)
    
# load the data and the network
train_loader, test_loader, label_dict = get_loaders(train_name, test_name, full_params, batch_size)
#CustomDataLoaders.plot_classes_histogram(train_loader, nums_to_labels(label_dict))

# print('accuracy {:.3f}'.format(trainer.mixed_model_validator(train_loader, model)))
# CustomDataLoaders.plot_classes_histogram(train_loader, nums_to_labels(label_dict))
print('calculating accuracy')
percent = True
confusion, accuracy = create_confusion_matrix(train_loader, model, label_dict, percent, custom_validation, splits)
print('accuracy {:.3f}'.format(accuracy))
plot_confusion(confusion, net_name, matrix_name)
# plot_confusion_percent(confusion, label_dict, net_name)

