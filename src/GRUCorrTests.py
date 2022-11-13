# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:30:39 2021

@author: gedadav
"""


from LivnatNetClassifier import ConvNet, NLayerGRU, BiGRU
import matplotlib.pyplot as plt
import RandomWalks
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop, ASGD, Adamax, AdamW, Adagrad, LBFGS
import DatasetCreator, DatasetCreatorTests
import torch.nn.functional as F
from CustomDataLoaders import MyDataset, StringLabelDataset
from torch.utils.data import DataLoader
from customoptimizers import ConstIntervalOptimizer
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



"""
the same GRU used for training so it will be able to load the weights. The only difference is that the forward method
also returns the hidden state and the outputs of each layer so we can use it to compare to the correlations
# regular rnn
"""
class AnalysisGRU(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Layer GRU', num_layers=1):
        super(AnalysisGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, hidden = self.gru(input.view(input.shape[0], input.shape[1], 1))
        X = ht[:, -1, :]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X, ht, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
# the bidirectional rnn
class AnalysisBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Bi Directional Layer GRU', num_layers=1):
        super(AnalysisBiGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, hidden = self.gru(input.view(input.shape[0], input.shape[1], 1))
        ht = ht.view(input.shape[0], input.shape[1], 2, self.hidden_size)
        X = torch.cat((ht[:,-1,0,:], ht[:,-1,1,:]), dim=1)
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X, ht, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


def load_data(label_name, param_name, network_file, net_type):
    with open(label_name, 'r') as f:
        labels = json.load(f)
    # label_dict = labels_to_nums(labels)
    with open(param_name, 'r') as res:
        full_params = json.load(res)
    # create a dictionary containing all the parameters of the experiment so it can be reproduced
    full_params.update(labels)
    
    if net_type is NLayerGRU:
        model = AnalysisGRU(1, full_params['hidden size'], full_params['names'], num_layers=full_params['layers'])
    elif net_type is ConvNet:
        model = ConvNet(full_params['walkers length'], full_params['names'], full_params['batch size'])
    elif net_type is BiGRU:
        model = AnalysisBiGRU(1, full_params['hidden size'], full_params['names'], num_layers=full_params['layers'])
    
    # then load the weights from the correct file
    saved_model = torch.load(network_file)
    # then set the network's weights accordingly
    model.load_state_dict(saved_model)
    return model, full_params
    

def get_loaders(train_datafile, test_datafile, full_params, batch_size):
    label_dict = full_params['label dictionary']
    train_data = StringLabelDataset(train_datafile, label_dict)    
    test_data = StringLabelDataset(test_datafile, label_dict)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, label_dict
    

def val(validation_loader, model):
    """
    returns:
        the validation set mean loss
        the validation set mean accuracy
    """
    model.eval()
    batch_size = validation_loader.batch_size
    with torch.no_grad():
        correct = 0
        v_loss = 0
        for data in validation_loader:
            # split into label and sample and run them through the network
            X, y = data
            output, _, _ = model(X)
            v_loss += F.nll_loss(output, y).item()
            pred = output.max(1, keepdim=True)[1]
            correct += int(pred.eq(y.view_as(pred)).cpu().sum())
    return 100 * (correct / (len(validation_loader)*int(batch_size)))
         

def labels_to_nums(labels):
    return {str(labels[i]): i for i in range(len(labels))}

def nums_to_labels(labels):
    return {num:label for label,num in labels.items()}

def setup_env(location, net_name):
    # Use seaborn style defaults and set the default figure size
    sns.set()
    # increase plot size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    
    # create the folder to save results
    full_location = location + '/' + net_name
    network_location = full_location + '/network/'
    hidden_states_loc = full_location + '/hidden states/'
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
        os.mkdir(hidden_states_loc)
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
    return network_location, hidden_states_loc, dataset_loc, param_loc


def hidden_states_to_numpy(loader, model, batch_size):
    walks = []
    with torch.no_grad():
        for data in loader:
            # split into label and sample and run them through the network
            X, y = data
            output, ht, hidden = model(X)
            # pred = output.max(1, keepdim=True)[1]
            list_ht = ht.detach().numpy()[:, -1, :].tolist()
            for i in range(y.shape[0]):
                walks.append(list_ht[i])
    return np.array(walks)


def plot_hidden_states(loader, model, walk_type, save_loc):
    """
    plots all the hidden states of the gru in a single plot. assuming the batch size is 1
    Parameters
    ----------
    loader : DataLoader
        the train loader (since its larger)
    model : model.nn
        a NN model (the GRU)
        
    Returns
    -------
    None.
    """
    plt.figure()
    model.eval()
    walks = hidden_states_to_numpy(loader, model , loader.batch_size)
    for walk in walks:
        plt.plot([t for t in range(int(walks.shape[1]))], walk)          
    plt.title('hidden states ' + walk_type)
    plt.savefig(save_loc + '.png')
    plt.show()
    return walks


def plot_bi_hidden_states(loader, model, walk_type, save_loc):
    """
    plots all the hidden states of the gru in a single plot. assuming the batch size is 1
    Parameters
    ----------
    loader : DataLoader
        the train loader (since its larger)
    model : model.nn
        a NN model (the GRU)
        
    Returns
    -------
    None.
    """
    model.eval()
    walks = hidden_states_to_numpy(loader, model, loader.batch_size)
    plt.figure()
    for walk in walks:
        plt.plot([t for t in range(walks.shape[2])], walk[0, :])          
    plt.title('hidden states forward ' + walk_type)
    plt.savefig(save_loc + '_forward.png')
    plt.show()
    plt.figure()
    for walk in walks:
        plt.plot([t for t in range(walks.shape[2])], walk[1, :])          
    plt.title('hidden states backward ' + walk_type)
    plt.savefig(save_loc + '_backward.png')
    plt.show()
    return walks
    
            

def plot_hiddenstate_ensemble_autocorr(loader, model, walk_type, steps, delta_t):
    plt.figure()
    model.eval()
    # create an array to hold the hidden states of the network as a set of random walks of shape (amount, steps)
    walks = hidden_states_to_numpy(loader, model, loader.batch_size)
    print('autocorrelation calculation starting')
    auto_corr = RandomWalks.ensemble_auto_corr_walks(walks, walks.shape[1], walks.shape[0])
    plt.plot(np.linspace(0, steps*delta_t, steps), auto_corr)
    plt.title('hidden state autocorrelation ' + walk_type)
    return auto_corr


def plot_walks_ensemble_autocorr(loader, model, walk_type, steps, delta_t):
    plt.figure()
    model.eval()
    # create an array to hold the hidden states of the network as a set of random walks of shape (amount, steps)
    walks = []
    with torch.no_grad():
        for data in loader:
            # split into label and sample and run them through the network
            X, y = data
            list_x = X.detach().numpy().tolist()
            for i in range(batch_size):
                walks.append(list_x[i])
    walks = np.array(walks)
    print('ensemble autocorrelation calculation starting')
    auto_corr = RandomWalks.ensemble_auto_corr_walks(walks, walks.shape[1], walks.shape[0])
    plt.plot(np.linspace(0, steps*delta_t, steps), auto_corr)
    plt.title('walks autocorrelation ' + walk_type)
    return auto_corr
    times = np.linspace(0, steps*delta_t, steps) 
    plt.plot(times, auto_corr)
    plt.title('ensemble autocorrelation ' + walk_type)
    
    
def plot_hiddenstate_autocorr(loader, model, walk_type, steps, delta_t):
    plt.figure()
    model.eval()
    # create an array to hold the hidden states of the network as a set of random walks of shape (amount, steps)
    walks = hidden_states_to_numpy(loader, model, loader.batch_size)
    print('autocorrelation calculation starting')
    auto_corr = RandomWalks.auto_corr_walks(walks)
    times = np.linspace(0, steps*delta_t, steps) 
    for walk in auto_corr:
        plt.plot(times, walk)
        plt.title('hidden state autocorrelation ' + walk_type)
    return auto_corr
    
    
def plot_hiddenstate_ensemble_tamsd(loader, model, walk_type, steps, delta_t):
    plt.figure()
    model.eval()
    # create an array to hold the hidden states of the network as a set of random walks of shape (amount, steps)
    walks = hidden_states_to_numpy(loader, model, loader.batch_size)
    print('TA_MSD calculation starting')
    ta_msd = RandomWalks.ensemble_TA_MSD(np.asarray(walks))
    times = np.linspace(0, steps*delta_t, steps) 
    plt.plot(times, ta_msd)
    plt.title('ta-msd ' + walk_type)
    return ta_msd
            
    
                                  
    
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

recreate_dataset = True
# recreate_dataset = False
# setup the all the locations and folders for the results to be placed
location = 'correlation tests'

#----------------GRU network tests
# net_name = 'NLayerGRU_mixed'
# net_name = 'NLayerGRU_non_mixed'
# net_type = NLayerGRU
#----------------BiGRU network tests
# net_name = 'BiGRU_mixed'
net_name = 'BiGRU_non_mixed'
net_type = BiGRU
#----------------Convolution network tests
# net_name = 'Convolution_non_mixed'
# net_name = 'Convolution_mixed'
# net_type = ConvNet

network_location, hidden_states_loc, dataset_loc, param_loc = setup_env(location, net_name)   
 # you first initialize the network you wish to use
network_location = network_location + 'net.pt'
label_name = param_loc + 'parameters.json'
param_name = param_loc + 'results.json'
fake_name = param_loc + 'fake_params.json'
# load all the parameters and the model from the files
model, full_params = load_data(label_name, param_name, network_location, net_type)
# check whether you classify the models that took part in the process or whether or not there has been a transition
classify_models = False if type(full_params['names'][0]) is bool else True
diffusion = UniformGenerator(0,1)
# diffusion = 0.3
delta_t = full_params['delta_t']
walkers = int(5e3)
steps = full_params['length']
train, test = [0.9, 0.1]
names = full_params['processes']
classify_diffusion = False
# the alpha of the waiting times power law distribution
alpha = full_params['alpha range']

batch_size = 64

for process in full_params['processes'].items():
    train_name, test_name = dataset_loc + 'train_{}.csv'.format(process[0]), dataset_loc + 'test_{}.csv'.format(process[0])
    creation_name = {process[0]: process[1]}
    if recreate_dataset:
        if full_params['model transition enabled']:
            train_set, test_set = DatasetCreator.createmixdataset(diffusion=diffusion, delta_t=delta_t, steps=steps,
                                      train_walkers=int(walkers*train), test_walkers=int(walkers*test),
                                      names=creation_name, train_name=train_name, test_name=test_name, classify_diffusion=classify_diffusion,
                                      max_transitions=2, transition_prob=1e-2, class_transition=classify_models, param_name=fake_name,
                                      alpha=alpha)
        else:
            train_set, test_set = DatasetCreator.create_dataset(diffusion=diffusion, delta_t=delta_t, steps=steps,
                                  train_walkers=int(walkers*train), test_walkers=int(walkers*test),
                                  names=creation_name, train_name=train_name, test_name=test_name, classify_diffusion=False,
                                  param_name=fake_name, alpha=alpha)    
        DatasetCreator.plot_dataset(test_set)
        
    # load the data and the network
    train_loader, test_loader, label_dict = get_loaders(train_name, test_name, full_params, batch_size)
    
    # CustomDataLoaders.plot_classes_histogram(train_loader, nums_to_labels(label_dict))
    # print the accuracy of the model on the given dataset
    print('accuracy over {}: {:.2f}%'.format(process[0], val(train_loader, model)))
    # define a location to save the plots in
    hidden_state_plot = hidden_states_loc + '{}'.format(process[0])
    """ plot all the hidden states of the network over the dataset """
    if net_type is NLayerGRU:
        hiddens = plot_hidden_states(train_loader, model, process[0], hidden_state_plot)
    elif net_type is BiGRU:
        hiddens = plot_bi_hidden_states(train_loader, model, process[0], hidden_state_plot)
    """plot the ensembled mean autocorrelation of the hidden states and the mean of the hidden states"""
    # autocorrs = plot_hiddenstate_ensemble_autocorr(train_loader, model, list(names.keys())[0], steps, delta_t)
    """plot ensemble autocorrelation of the random walks """
    # walk_autocorrs = plot_walks_ensemble_autocorr(train_loader, model, list(names.keys())[0], steps, delta_t)
    # # plot both the hidden states and the ensemble autocorrelation of the hidden states on the same plot
    # plt.figure()
    # plt.plot(np.linspace(0, steps*delta_t, steps), walk_autocorrs, label='walks ensemble autocorrelation')
    # plt.plot(np.linspace(0, steps*delta_t, steps), autocorrs, label='hidden state ensemble autocorrelation')
    # plt.legend()
    """ plot the regular autocorrelation of the hidden states"""
    # plot_hiddenstate_autocorr(train_loader, model, list(names.keys())[0], steps, delta_t)
    """ plot the ensembled mean ta_msd of the hidden states """
    # plot_hiddenstate_ensemble_tamsd(train_loader, model, list(names.keys())[0], steps, delta_t)
    # plt.figure()
    # plt.plot(np.linspace(0, steps*delta_t, steps), 5 * np.mean(hiddens, axis=0), label='mean hidden states')
    # plt.plot(np.linspace(0, steps*delta_t, steps), autocorrs, label='ensemble autocorrelation')
    # plt.legend()

# # take some sample from the test set
# iterator = iter(test_loader)
# x, y = next(iterator)
# # keep sampling until you get a batch with regular brownian motion
# # while 0 not in y:
# #     x, y = next(iterator)
# # get the prediction on the sampled input
# output, ht, hidden = model(x)
# # gets the highest likelyhood prediction (in this case the index in the one how vector of classification)
# pred = output.max(1, keepdim=True)[1]
# # returns a boolean vector of the correctnes of predictions
# correct = pred.eq(y.view_as(pred))
# # get a brownian random walk that the network recognizes
# for index in range(batch_size):
#     if y[index] == label_dict['Brownian'] and correct[index]:
#         break
# # get a FBM walk that the network recognizes
# for index2 in range(batch_size):
#     if y[index2] == label_dict['FBM'] and correct[index2]:
#         break

# # take the output of GRU at time t for the sample we got at index
# out1 = ht.detach().numpy()[index, -1, :]
# # take the hidden state of the final layer of the GRU at time t
# # note that the for the final layer, the hidden states are the same as the output
# hidden1 = hidden.detach().numpy()[1, index, :]
# # take the output of GRU at time t for the sample we got at index
# out2 = ht.detach().numpy()[index2, -1, :]
# # take the hidden state of the final layer of the GRU at time t
# # note that the for the final layer, the hidden states are the same as the output
# hidden2 = hidden.detach().numpy()[1, index2, :]
# # plot both as a function of t
# times = [i for i in range(steps)]
# plt.figure()
# plt.plot(times, out1, label='Brownian')
# # plt.plot(times, out2, label='FBM')
# # plt.plot(times, hidden, label='second layer hidden')
# # # also plot the random walk itself
# walk1 = x.detach().numpy()[index, :]
# plt.plot(times, walk1, label='brownian random walk')
# # walk2 = x.detach().numpy()[index2, :]
# # plt.plot(times, walk2, label='FBM random walk')
# # # plot the pearson correlation function of the walk
# # pearson_autocorr = RandomWalks.pearson_auto_corr_function(walk)
# # plt.plot(times, pearson_autocorr, label='pearson correlation')
# # # plot the convolution correlation
# # conv_autocorr = RandomWalks.conv_correlation(walk, walk)
# # plt.plot(times, conv_autocorr, label='convolution correlation')
# # # plot numpy correlation
# # np_autocorr = np.correlate(walk, walk, mode='full')[steps-1:]
# # plt.plot(times, np_autocorr, label='numpy correlation')
# plt.legend()







