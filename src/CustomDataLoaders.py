# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:39:59 2020

@author: gedadav
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval



# a dataset loader without the inclusion of sklearn scaler
class MyDataset(Dataset):  
    def __init__(self, csv_file):
        # loads an indexed data file that was saved using pandas DataFrame, so we skip the first column
        xy = pd.read_csv(csv_file).iloc[:, 1:]
        xy = xy.to_numpy()
        self.x = torch.from_numpy(xy[:, :-1]).float()
        self.y = torch.from_numpy(xy[:, -1]).float()
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    
# a dataset loader without the inclusion of sklearn scaler
class StringLabelDataset(Dataset):  
    def __init__(self, csv_file, label_dict=None, class_by='model'):
        # loads an indexed data file that was saved using pandas DataFrame, so we skip the first column
        xy = pd.read_csv(csv_file).iloc[:, 1:]
        # drop all columns except the ones that we will classify by and change the column name to label
        xy = self.drop_columns(xy, class_by).rename(columns={class_by: 'label'})
        xy, self.label_dict = self.change_to_numeric(xy, label_dict)
        xy = xy.to_numpy()
        # TODO change back
        # xy = np.absolute(xy.to_numpy())
        self.x = torch.from_numpy(xy[:, :-1].astype('float32'))
        self.y = torch.from_numpy(xy[:, -1].astype('float32')).long()
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    def get_label_dict(self):
        return self.label_dict.copy()
    
    def change_to_numeric(self, data, label_dict):
        labels = list(set(data.label))
        if label_dict is None:
            label_dict = {labels[i]:i for i in range(len(labels))}
        for label in label_dict.keys():
            data.loc[data.label == label, 'label'] = label_dict[label]
        return data, label_dict
    
    def drop_columns(self, df, classification):
        for col in df.copy().columns:
            if 'step' in col or col == classification:
                continue
            else:
                df = df.drop(columns=[col])
        return df


class MixedModelDataset(Dataset):
    def __init__(self, csv_file, label_dict, class_by='model'):
        # loads an indexed data file that was saved using pandas DataFrame, so we skip the first column
        xy = pd.read_csv(csv_file).iloc[:, 1:]
        # convert the model column into a list
        xy['model'] = xy['model'].apply(literal_eval)
        # split the label column into the two constructing models
        xy['label1'] = xy['model'].apply(lambda x : label_dict[x[0]])
        xy['label2'] = xy['model'].apply(lambda x : label_dict[x[1]])
        # drop all columns except the ones that we will classify by and change the column name to label
        xy = self.drop_columns(xy, classification=['label1', 'label2'])
        self.label_dict = label_dict
        xy = xy.to_numpy()
        self.x = torch.from_numpy(xy[:, :-2].astype('float32'))
        self.y = torch.from_numpy(xy[:, -2:].astype('float32')).long()
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    def get_label_dict(self):
        return self.label_dict.copy()
    
    def drop_columns(self, df, classification):
        for col in df.copy().columns:
            if 'step' in col or col == classification[0] or col == classification[1]:
                continue
            else:
                df = df.drop(columns=[col])
        return df


def plot_classes_histogram(loader, labels):
    """
    this function plots a histogram of the data distribution
    Parameters
    ----------
    loader : DataLoader
        a data loader for the histogram
    labels : dict
        a dictionary that maps numbers to the labels used

    Returns
    -------
    None.
    """
    # create a histogram of all the classes
    fig, ax = plt.subplots()
    ax.set_xticks([])
    classes = []
    with torch.no_grad():
        for data in loader:
            # split into label and sample and run them through the network
            X, y = data
            list_y = y.detach().numpy().tolist()
            classes.extend(list_y)
    counts, bins, _ = plt.hist(classes, bins=len(list(labels.keys())))
    # # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    temp_list = list(counts).copy()
    for count, x in zip(counts, bin_centers):
        # # Label the raw counts
        # ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        #             xytext=(0, -18), textcoords='offset points', va='top', ha='center')
        # the following step is to make sure same value objects arent chosen twice
        ind = temp_list.index(count)
        temp_list[ind] = None
        # find the name of the label that this bin belongs to and add it to the list
        name = labels[ind]
        ax.annotate(name, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -5), textcoords='offset points', va='top', ha='center', rotation='vertical')


    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    
"""
# create a dataset from your csv file
dataset = MyDataset('const diffusion 100steps 1000walkers.csv')

features, labels = dataset[0]
print(features)
print(labels)

# create a dataloader from the dataset you created.
loader = DataLoader(dataset, batch_size=10, shuffle=True)
# check that its all good
dataiter = iter(loader)
feature, labels = dataiter.next()
print(feature[1])
print(labels[1])
"""