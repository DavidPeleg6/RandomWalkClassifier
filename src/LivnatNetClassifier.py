# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:50:37 2020

@author: gedadav
"""

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop, ASGD, Adamax, AdamW, Adagrad, LBFGS
import torch.nn.functional as F
from CustomDataLoaders import MyDataset, StringLabelDataset
from torch.utils.data import DataLoader
from trainer import train, val, regression_train, regression_val, special_val
from customoptimizers import ConstIntervalOptimizer
from Generator import NormalGenerator
import time
from Plotter import TwoValPlotter
import os
from datetime import datetime
import sys
import json
from RandomWalks import TA_MSD, auto_corr_walks
from torch.autograd import Variable


#---------------------------HYPER PARAMETERS---------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#name = os.path.splitext(os.path.basename(__file__))[0] + f'epoch {EPOCHS} time'
name = None


def save_results(location, filename, results, model, optimizer):
    try:
        os.mkdir(location)
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{location}/{filename}')
    except FileExistsError:
        pass
    result_file = f'{location}/{filename}/results.json'
    with open(result_file, 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=True)
    torch.save(model.state_dict(), f'{location}/{filename}/net.pt')
    torch.save(optimizer.state_dict(), f'{location}/{filename}/optimizer.pt')


class ConvNet(nn.Module):
    def __init__(self, walk_length, labels, batch_size, name='Convolutional NN'):
        super(ConvNet, self).__init__()
        self.name = name
        self.n = walk_length
        self.batch_size = batch_size
        #self.norm = nn.LayerNorm([walk_length])
        # every sample has 1 random walk meaning 1 input channel
        # we would like to operate with 16 filters meaning 16 out channels
        # were taking the nearest neighbot so were going with a 2 convolution
        # dilation = spacing between kernel elements. when we take next nearest neighbor it will be 2
        # dilation simulation https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.conv1 = nn.Conv1d(1, out_channels=16, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(1, out_channels=16, kernel_size=2, dilation=2)
        self.conv3 = nn.Conv1d(1, out_channels=16, kernel_size=2, dilation=3)
        #self.drop1 = nn.Dropout(p=0.2)
        #self.drop2 = nn.Dropout(p=0.2)
        # calculating size according to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        fc_size = (3 * self.n - 6) * 16
        self.fc1 = nn.Linear(fc_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, len(labels))
        
    def forward(self, x):
        # change dimensions and type to match the convolution layer
        #x = self.norm(x)
        walk_length = self.n
        batch_size = x.shape[0]
        X = x.view(batch_size, 1, walk_length).float()
        # a custom activation function that squares and normalizes the additions made in the kernel
        # TODO change these
        # x1 = (1/self.n) * torch.square(self.conv1(X))
        # x2 = (1/self.n) * torch.square(self.conv2(X))
        # x3 = (1/self.n) * torch.square(self.conv3(X))
        x1 = F.relu(self.conv1(X))
        x2 = F.relu(self.conv2(X))
        x3 = F.relu(self.conv3(X))
        # flatten out and concatenate the outputs of the 3 convolutions
        X = torch.cat((x1.view(batch_size, -1), x2.view(batch_size, -1), x3.view(batch_size, -1)), 1)
        # fully connected part
        X = F.relu(self.fc1(X))
        #X = self.drop1(X)
        X = F.relu(self.fc2(X))
        #X = self.drop2(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)
    

class BasicNet(nn.Module):
    def __init__(self, walk_length, labels, name='Fully Connected NN'):
        super(BasicNet, self).__init__()
        self.name = name
        #self.drop1 = nn.Dropout(p=0.2)
        #self.drop2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(walk_length, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, len(labels))
        
    def forward(self, x):
        X = F.relu(self.fc1(x))
        #X = self.drop1(X)
        X = F.relu(self.fc2(X))
        #X = self.drop2(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)
        
       
class NLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Layer RNN', num_layers=1):
        super(NLayerRNN, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu', batch_first=True)
        #self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, hidden = self.rnn(input.view(input.shape[0], input.shape[1], 1))
        X = ht[:, -1, :]
        #X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    
class NLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Layer LSTM', num_layers=1):
        super(NLayerLSTM, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, _ = self.lstm(input.view(input.shape[0], input.shape[1], 1))
        X = ht[:, -1, :]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    

class NLayerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Layer GRU', num_layers=1):
        super(NLayerGRU, self).__init__()     
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
        # if the data is given as a packed sequence it should not be reshaped
        if type(input) != torch.nn.utils.rnn.PackedSequence:
            input = input.view(input.shape[0], input.shape[1], 1)
        ht, hidden = self.gru(input)
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            ht, ht_lens = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        X = ht[:, -1, :]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class ConvolutionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Layer ConvolutionalGRU', num_layers=1, encoders=1):
        super(ConvolutionalGRU, self).__init__()
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_channels = 64
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, out_channels=self.out_channels, kernel_size=10),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            # nn.Conv1d(10, out_channels=20, kernel_size=5),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(3)
        )
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.conv_gru = nn.GRU(self.out_channels, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(3 * hidden_size, hidden_size)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        # todo add a residual gru here that will take the regular input data and append it to the output of the convolution part
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        batch_size = input.shape[0]
        walk_length = input.shape[1]
        X = input.view(batch_size, 1, walk_length).float()
        # a custom activation function that squares and normalizes the additions made in the kernel
        # X = F.relu(self.pool1(self.conv1(X)))
        # X = F.relu(self.pool2(self.conv2(X)))
        X = self.conv_layers(X)
        # todo uncomment to go back to previous conv implementation
        # X = X.view(batch_size, -1, 1)
        X = X.transpose(1, 2)
        ht1, hidden1 = self.conv_gru(X)
        ht2, hidden2 = self.gru(input.view(batch_size, walk_length, 1))
        # todo probably delete
        # if type(X) == torch.nn.utils.rnn.PackedSequence:
        #     ht, ht_lens = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        X = torch.cat([ht1[:, -1, :], ht2[:, -1, :]], dim=1)
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class NLayerGRU2(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Layer GRU', num_layers=1):
        super(NLayerGRU, self).__init__()     
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
        # if the data is given as a packed sequence it should not be reshaped
        if type(input) != torch.nn.utils.rnn.PackedSequence:
            input = input.view(input.shape[0], input.shape[1], 1)
        ht, hidden = self.gru(input)
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            ht, ht_lens = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        X = ht[:, -1, :]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, labels, name='Bi Directional Layer GRU', num_layers=1):
        super(BiGRU, self).__init__()     
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
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    
class SpecialNLayerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, labels, preprocesses, name='walk processing GRU', num_layers=1):
        super(SpecialNLayerGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pre = preprocesses
        self.input_size = input_size + len(preprocesses)
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        X = self.apply_preprocess(input)
        ht, hidden = self.gru(X.view(input.shape[0], input.shape[1], self.input_size))
        X = ht[:, -1, :]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    # this method takes the inserts the hidden states between every value of the original walk: x' = [x[0], autocor(x[0]), ..., x[n], autocorr(x[n])]
    def apply_preprocess(self, X):
        # create a vector of X and preprocessed X
        new_X = torch.zeros(X.shape[0], self.input_size * X.shape[1])
        for col in range(X.shape[1]):
            new_X[:, col*self.input_size] = X[:, col]
        for i in range(len(self.pre)):
            temp = self.pre[i](X)
            for col in range(i+1, X.shape[1]):
                new_X[:, col*self.input_size] = torch.from_numpy(temp[:, col])
        return new_X
    
    
class SpecialNLayerGRU2(nn.Module):
    def __init__(self, input_size, hidden_size, labels, preprocesses, name='hidden state processing GRU', num_layers=1):
        super(SpecialNLayerGRU2, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pre = preprocesses
        self.input_size = input_size
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * (len(preprocesses) + 1), hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, hidden = self.gru(input.view(input.shape[0], input.shape[1], self.input_size))
        X = self.apply_preprocess(ht[:, -1, :])
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    # this method applies 
    def apply_preprocess(self, X):
        # create a vector of X and preprocessed X
        new_X = torch.zeros(X.shape[0], (len(self.pre) + 1) * X.shape[1])
        new_X[:, :X.shape[1]] = X
        for i in range(1, len(self.pre)+1):
            temp = self.pre[i-1](X.detach().numpy())
            new_X[:, X.shape[1]*i:X.shape[1]*(i+1)] = torch.from_numpy(temp.copy())
        return new_X
    

class SingleStepGRU(nn.Module):
    """
    this network is the same as the regular layered GRU only this time there is only one hidden state for each time step
    and the FC layer takes the entire h_t vector into account for classification. Therfor, it will need to know the length of the random walks
    """
    def __init__(self, input_size, walk_length, labels, name='Single Step GRU', num_layers=1):
        super(SingleStepGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.num_layers = num_layers
        self.n = walk_length
        # self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, 1, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(walk_length, walk_length)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(walk_length, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, hidden = self.gru(input.view(input.shape[0], input.shape[1], 1))
        X = ht[:, :, 0]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class BiSingleStepGRU(nn.Module):
    """
    this network is the same as the regular layered GRU only this time there is only one hidden state for each time step
    and the FC layer takes the entire h_t vector into account for classification. Therfor, it will need to know the length of the random walks
    """
    def __init__(self, input_size, walk_length, labels, name='Bi Directional Single Step GRU', num_layers=1):
        super(BiSingleStepGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.num_layers = num_layers
        self.n = walk_length
        # self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, 1, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * walk_length, walk_length)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(walk_length, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, hidden = self.gru(input.view(input.shape[0], input.shape[1], 1))
        ht = ht.view(input.shape[0], input.shape[1], 2, 1)
        X = torch.cat((ht[:,:,0,0], ht[:,:,1,0]), dim=1)
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    

class SpecialNLayerGRU3(nn.Module):
    def __init__(self, input_size, hidden_size, labels, preprocesses, name='walk concat processing GRU', num_layers=1):
        super(SpecialNLayerGRU3, self).__init__()     
        self.name = f'{num_layers} ' + name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pre = preprocesses
        self.input_size = input_size + len(preprocesses)
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        X = self.apply_preprocess(input)
        ht, hidden = self.gru(X.view(input.shape[0], input.shape[1], self.input_size))
        X = ht[:, -1, :]
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    
    def apply_preprocess(self, X):
        # create a vector of X and preprocessed X
        new_X = torch.zeros(X.shape[0], self.input_size * X.shape[1])
        new_X[:, :X.shape[1]] = X
        # repeat the process for all the preprocessing functions given
        for i in range(1, len(self.pre)+1):
            temp = self.pre[i-1](X.detach().numpy())
            new_X[:, X.shape[1]*i:X.shape[1]*(i+1)] = torch.from_numpy(temp.copy())
        return new_X
              
    
class TransitionGRU(nn.Module):
    def __init__(self, input_size, walk_length, hidden_size, labels, name='Layer GRU', num_layers=1):
        super(TransitionGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        # for the 2D case we might want to use hidden size=2
        self.hidden_size = hidden_size
        self.walk_length = walk_length
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear((walk_length) * hidden_size, walk_length)
        self.drop = nn.Dropout(0.1)
        # not taking into account the first step = ignoring the value of the first hidden state
        self.fc2 = nn.Linear(walk_length, len(labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, _ = self.lstm(input.view(input.shape[0], input.shape[1], 1))
        X = ht[:, :, :].reshape((input.shape[0], (self.walk_length) * self.hidden_size))
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(self.drop(X)))
        X = self.softmax(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    
class RegressionGRU(nn.Module):
    def __init__(self, input_size, walk_length, hidden_size, labels, name='Layer GRU', num_layers=1):
        super(RegressionGRU, self).__init__()     
        self.name = f'{num_layers} ' + name
        # for the 2D case we might want to use hidden size=2
        self.hidden_size = hidden_size
        self.walk_length = walk_length
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        #self.fc1 = nn.Linear((walk_length) * hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        ht, _ = self.gru(input.view(input.shape[0], input.shape[1], 1))
        # X = ht[:, :, :].reshape((input.shape[0], (self.walk_length) * self.hidden_size))
        X = ht[:, -1, :].reshape((input.shape[0], self.hidden_size))
        X = F.relu(self.fc1(X))
        X = self.drop1(X)
        X = F.relu(self.fc2(X))
        X = self.drop2(X)
        X = self.fc3(X)
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    

class RegressionLSTM(nn.Module):
    def __init__(self, input_size, walk_length, hidden_size, labels, name='Layer LSTM', num_layers=1):
        super(RegressionLSTM, self).__init__()     
        self.name = f'{num_layers} ' + name
        # for the 2D case we might want to use hidden size=2
        self.hidden_size = hidden_size
        self.walk_length = walk_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        #self.fc1 = nn.Linear(walk_length, walk_length)
        # not taking into account the first step = ignoring the value of the first hidden state
        self.fc2 = nn.Linear((walk_length) * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        ht, _ = self.lstm(input.view(input.shape[0], input.shape[1], 1))
        X = ht[:, :, :].reshape((input.shape[0], (self.walk_length) * self.hidden_size))
        #X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        return X

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    
class RegressionConvNet(nn.Module):
    def __init__(self, walk_length, labels, batch_size, name='Convolutional NN'):
        super(RegressionConvNet, self).__init__()
        self.name = name
        self.n = walk_length
        self.batch_size = batch_size
        #self.norm = nn.LayerNorm([walk_length])
        # every sample has 1 random walk meaning 1 input channel
        # we would like to operate with 16 filters meaning 16 out channels
        # were taking the nearest neighbot so were going with a 2 convolution
        # dilation = spacing between kernel elements. when we take next nearest neighbor it will be 2
        # dilation simulation https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.conv1 = nn.Conv1d(1, out_channels=16, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(1, out_channels=16, kernel_size=2, dilation=2)
        self.conv3 = nn.Conv1d(1, out_channels=16, kernel_size=2, dilation=3)
        #self.drop1 = nn.Dropout(p=0.2)
        #self.drop2 = nn.Dropout(p=0.2)
        # calculating size according to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        fc_size = (3 * self.n - 6) * 16
        self.fc1 = nn.Linear(fc_size, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 1)
        #self.fc4 = nn.Linear(30, len(labels))
        
    def forward(self, x):
        # change dimensions and type to match the convolution layer
        #x = self.norm(x)
        walk_length = self.n
        batch_size = x.shape[0]
        X = x.view(batch_size, 1, walk_length).float()
        # a custom activation function that squares and normalizes the additions made in the kernel
        # TODO change these
        # x1 = (1/self.n) * torch.square(self.conv1(X))
        # x2 = (1/self.n) * torch.square(self.conv2(X))
        # x3 = (1/self.n) * torch.square(self.conv3(X))
        x1 = F.relu(self.conv1(X))
        x2 = F.relu(self.conv2(X))
        x3 = F.relu(self.conv3(X))
        # flatten out and concatenate the outputs of the 3 convolutions
        X = torch.cat((x1.view(batch_size,-1), x2.view(batch_size,-1), x3.view(batch_size,-1)), 1)
        # fully connected part
        X = F.relu(self.fc1(X))
        #X = self.drop1(X)
        X = F.relu(self.fc2(X))
        #X = self.drop2(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return X
    
    
class BasicRegression(nn.Module):
    def __init__(self, walk_length, name='Fully Connected regression NN'):
        super(BasicRegression, self).__init__()
        self.name = name
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(walk_length, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 1)
        
    def forward(self, x):
        X = F.relu(self.fc1(x))
        X = self.drop1(X)
        X = F.relu(self.fc2(X))
        X = self.drop2(X)
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        return X
    
    
class WideConvNet(nn.Module):
    def __init__(self, walk_length, labels, batch_size, name='Convolutional NN'):
        super(WideConvNet, self).__init__()
        self.name = name
        self.n = walk_length
        self.batch_size = batch_size
        #self.norm = nn.LayerNorm([walk_length])
        # every sample has 1 random walk meaning 1 input channel
        # we would like to operate with 16 filters meaning 16 out channels
        # were taking the nearest neighbot so were going with a 2 convolution
        # dilation = spacing between kernel elements. when we take next nearest neighbor it will be 2
        # dilation simulation https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.conv1 = nn.Conv1d(1, out_channels=64, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(1, out_channels=64, kernel_size=2, dilation=2)
        self.conv3 = nn.Conv1d(1, out_channels=64, kernel_size=2, dilation=3)
        #self.drop1 = nn.Dropout(p=0.2)
        #self.drop2 = nn.Dropout(p=0.2)
        # calculating size according to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        fc_size = (3 * self.n - 6) * 64
        self.fc1 = nn.Linear(fc_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, len(labels))
        
    def forward(self, x):
        # change dimensions and type to match the convolution layer
        #x = self.norm(x)
        walk_length = self.n
        batch_size = x.shape[0]
        X = x.view(batch_size, 1, walk_length).float()
        # a custom activation function that squares and normalizes the additions made in the kernel
        # TODO change these
        # x1 = (1/self.n) * torch.square(self.conv1(X))
        # x2 = (1/self.n) * torch.square(self.conv2(X))
        # x3 = (1/self.n) * torch.square(self.conv3(X))
        x1 = F.relu(self.conv1(X))
        x2 = F.relu(self.conv2(X))
        x3 = F.relu(self.conv3(X))
        # flatten out and concatenate the outputs of the 3 convolutions
        X = torch.cat((x1.view(batch_size,-1), x2.view(batch_size,-1), x3.view(batch_size,-1)), 1)
        # fully connected part
        X = F.relu(self.fc1(X))
        #X = self.drop1(X)
        X = F.relu(self.fc2(X))
        #X = self.drop2(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)


class WideBasicNet(nn.Module):
    def __init__(self, walk_length, labels, name='Fully Connected NN'):
        super(WideBasicNet, self).__init__()
        self.name = name
        #self.drop1 = nn.Dropout(p=0.2)
        #self.drop2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(walk_length, 700)
        self.fc2 = nn.Linear(700, 500)
        self.fc2_1 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc3_1 = nn.Linear(100, 30)
        self.fc4 = nn.Linear(30, len(labels))
        
    def forward(self, x):
        X = F.relu(self.fc1(x))
        #X = self.drop1(X)
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc2_1(X))
        #X = self.drop2(X)
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc3_1(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)
        


def main_loop(train_datafile, test_datafile, exper_data, full_data_location, param_file, hidden_size=0, layers=1, location='', regression=False):
    total_t = time.time()
    # setting up all the variables needed from the experiment data dictionary
    net_type = exper_data['net type']
    optimizer_type = exper_data['optimizer']
    walk_length = exper_data['walkers length']
    batch_size = exper_data['batch size']
    alpha_0 = exper_data['learning rate']
    EPOCHS = exper_data['epochs']
    # a list of preprocessing functions
    preprocess = [globals()[pre] for pre in exper_data['pre-process']]
    # preprocess = exper_data['pre-process']
    #-----------------------------LOADING THE DATA--------------------------------
    if regression:
        train_data = MyDataset(train_datafile, preprocess)    
        test_data = MyDataset(test_datafile, preprocess)
    else:
        with open(param_file, 'r') as f:
            labels = json.load(f)['names']
        train_data = StringLabelDataset(train_datafile, class_by=exper_data['classify by'])    
        test_data = StringLabelDataset(test_datafile, class_by=exper_data['classify by'], label_dict=train_data.get_label_dict())
        exper_data['label dictionary'] = train_data.get_label_dict()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    #------------------------CREATING THE NETWORK INSTANCES-----------------------
    trainer, validator = [train, val] if not regression else [regression_train, regression_val]
    if net_type == 'Convolutional_NN':   
        net = ConvNet(walk_length, labels, batch_size).to(device)
    elif net_type == 'Fully_Connected_NN':
        net = BasicNet(walk_length, labels).to(device)
    elif net_type == 'vanilla_RNN':
        net = NLayerRNN(1, hidden_size, labels).to(device)
    elif net_type == 'N_layer_RNN':
        net = NLayerRNN(1, hidden_size, labels, num_layers=layers).to(device)
    elif net_type == 'N_layer_LSTM':
        net = NLayerLSTM(1, hidden_size, labels, num_layers=layers).to(device)
    elif net_type == 'N_layer_GRU':
        net = NLayerGRU(1, hidden_size, labels, num_layers=layers).to(device)
    elif net_type == 'ConvolutionalGRU':
        net = ConvolutionalGRU(1, hidden_size, labels, num_layers=layers, encoders=layers).to(device)
    elif net_type == 'Single_step_GRU':
        net = NLayerGRU(1, walk_length, labels, num_layers=layers).to(device)
    elif net_type == 'Bi_Single_step_GRU':
        net = NLayerGRU(1, walk_length, labels, num_layers=layers).to(device)
    elif net_type == 'Bi_Directional_GRU':
        net = BiGRU(1, hidden_size, labels, num_layers=layers).to(device)        
    elif net_type == 'Special_N_layer_GRU':
        net = SpecialNLayerGRU(1, hidden_size, labels, preprocess, num_layers=layers).to(device)
    elif net_type == 'Special_N_layer_GRU2':
        net = SpecialNLayerGRU2(1, hidden_size, labels, preprocess, num_layers=layers).to(device)
    elif net_type == 'Special_N_layer_GRU3':
        net = SpecialNLayerGRU3(1, hidden_size, labels, preprocess, num_layers=layers).to(device)
    elif net_type == 'Transition_GRU':
        net = TransitionGRU(1, walk_length, hidden_size, labels, num_layers=layers).to(device)
    elif net_type == 'Regression_GRU':
        net = RegressionGRU(1, walk_length, hidden_size, labels, num_layers=layers).to(device)
    elif net_type == 'Regression_LSTM':
        net = RegressionLSTM(1, walk_length, hidden_size, labels, num_layers=layers).to(device)
    elif net_type == 'Regression_Convolution':
        net = RegressionConvNet(walk_length, labels, batch_size).to(device)
    elif net_type == 'Regression_Fully_Connected':
        net = BasicRegression(walk_length).to(device)
    elif net_type == 'Wide_Convolution':
        net = ConvNet(walk_length, labels, batch_size).to(device)
    elif net_type == 'Wide_Fully_Connected':
        net = BasicNet(walk_length, labels).to(device)

        
    #----------------------CREATING THE OPTIMIZER------------------------------
    if optimizer_type == 'SGD':   
        optimizer = SGD(net.parameters(), alpha_0)
    if optimizer_type == 'Adam':
        optimizer = Adam(net.parameters(), alpha_0)
    if optimizer_type == 'RMSprop':
        optimizer = RMSprop(net.parameters(), alpha_0)
    if optimizer_type == 'ASGD':
        optimizer = ASGD(net.parameters(), alpha_0)
    if optimizer_type == 'Adamax':
        optimizer = Adamax(net.parameters(), alpha_0)
    if optimizer_type == 'AdamW':
        optimizer = AdamW(net.parameters(), alpha_0)
    if optimizer_type == 'Adagrad':
        optimizer = Adagrad(net.parameters(), alpha_0)
    if optimizer_type == 'LBFGS':
        optimizer = LBFGS(net.parameters(), alpha_0)
    
    #loss_plot = TwoValPlotter(title=f'{net.name} optimizer {optimizer_type}', y='Loss', label1='train', label2='validation')
    exper_title = f'{net.name} optimizer {optimizer_type}'
    save_folder = f'{full_data_location}{exper_title}'
    if regression:
        plotter = TwoValPlotter(title=exper_title, y='MAE loss', label1='train', label2='validation')
    else:
        plotter = TwoValPlotter(title=exper_title, y='accuracy', label1='train', label2='validation')
        
    #-------------------------------MAIN------------------------------------------
    for e in range(EPOCHS):
        t = time.time()
        train_avg_loss, train_avg_acc = trainer(train_loader, net, optimizer)    
        val_avg_loss, val_avg_acc = validator(test_loader, net) 
        
        #loss_plot.update(train_avg_loss, val_avg_loss)
        plotter.update(train_avg_acc, val_avg_acc)
        print("Epoch: {}/{}".format(e+1, EPOCHS),
              "Time in min: {:.3f}".format((time.time() - t)/60),
              "{}: {}".format(plotter.y, val_avg_acc))
        # prepare and save data for later analysis
        exper_data['current epoch'] = e
        exper_data[plotter.y] = val_avg_acc
        exper_data['runtime'] = round((time.time() - total_t) / 60, 3)
        save_results(save_folder, exper_data['start time'], exper_data, net, optimizer)
        

       
    print('loss: {} {}: {}'.format(val_avg_loss, plotter.y, val_avg_acc))
    print("total time", (time.time() - total_t)/60)
    #loss_plot.plot(name)
    plotter.plot(location)
    return net, plotter, test_loader, val_avg_acc
    
    
def labels_to_nums(labels):
    return {str(labels[i]): i for i in range(len(labels))}

def nums_to_labels(labels):
    return {i: labels[i] for i in range(len(labels))}


if __name__=='__main__':
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    