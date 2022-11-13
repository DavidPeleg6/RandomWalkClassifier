# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:25:40 2020

@author: gedadav
"""

from Generator import UniformGenerator, NormalGenerator, SymmetricExponGenerator, ExponGenerator, LevyGenerator, PowerLawGenerator
from RandomWalks import BasicRandomWalk, CTRW, FBM
import pandas as pd
import numpy as np
import json
import random
import itertools
import RandomWalks
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt


"""
using the TA-MSD method in RandomWalks.py to calculate the TA-MSD row by row
"""
def TA_MSD(df):
    new_set = df.copy()
    # iterate over each row and calculate its ta-msd
    for index, row in df.iterrows():
        # save the ta-msd vector into the new dataset
        new_set.iloc[index, :-1] = pd.DataFrame(RandomWalks.TA_MSD(row[:-1].to_numpy()))
    return new_set


# shouldnt be used since it treats each time step as a different feature and scales it. which means 
def scale_data(df, labels, scaler=None):
    """
    using sklearn preprocessing StandardScaler to scale the dataset.

    Parameters
    ----------
    df : DataFrame
        the entire dataset
    scaler : StandardScaler, optional
        the scaler that has already been used on the data. if None, will define a new data scaler

    Returns
    -------
    df : DataFrame
        the entire dataset, scaled
    scaler : StandardScaler
        the scaler used on the data so it can be reused for the test set
    """
    new_set = df.drop(labels, axis=1).to_numpy()
    if not scaler:
        # scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = preprocessing.Normalizer()
        X = scaler.fit_transform(new_set[:, :])
    else:
        X = scaler.transform(new_set[:, :])
    new_set[:, :] = X
    new_set = pd.DataFrame(new_set)
    new_set[labels] = df.loc[:, labels]
    new_set.columns = df.columns
    return new_set, scaler

def autocorrelate(df):
    new_set = df.copy()
    # save the autocorrelation vector into the new dataset
    new_set.iloc[:, :-1] = pd.DataFrame(RandomWalks.auto_corr_walks(new_set[:, :-1]))
    return new_set


# def normalize_walk(walk):
#     pass


# def normalize_dataset(df, label_names):
#     new_set = df.drop(label_names, axis=1)
#     new_set = new_set.div(new_set.sum(axis=1), axis=0)
#     df.loc[:, ~df.columns.isin(label_names)] = new_set
#     return df

        
def plot_dataset(set_name, save_loc='', labels='model'):
    global data_set
    data_set = pd.read_csv(set_name)
    # for plotting the testset
    for label in set(data_set[labels]):
        plt.figure()
        tempset = data_set[data_set[labels] == label]
        for walk in tempset.iloc[:, 1:-5].to_numpy():
            plt.plot([i for i in range(len(walk))], walk)
        plt.xlabel('t')
        plt.ylabel('x')
        # plt.title(label)
        plt.savefig(save_loc + label)
        plt.show()
             

def createprocesses(diffusion, delta_t, steps, names, const_diffusion=False, hurst_neg=0.3, 
                    hurst_pos=0.7, uni_time_gen=UniformGenerator(0,1), alpha=(2,3), levy_alpha=1.75):
    """
    creates a list of random walk creators with a smaller set of classes
    """
    #---------------------- create a list of processes and names -----------------
    processes = []
    # take the mean of the alpha range given for the constant alpha case
    mean_alpha = (alpha[0] + alpha[1]) / 2 if type(alpha) != float else alpha
    for key in names.keys():
        if 'Normal noise constant interval' == key:    
            processes.append({'generator': BasicRandomWalk(diffusion, delta_t, NormalGenerator(), steps, const_diffusion),
                              'name': names[key]})
        elif 'symmetric exponential noise constant interval' == key:    
            processes.append({'generator': BasicRandomWalk(diffusion, delta_t, SymmetricExponGenerator(), steps, const_diffusion),
                              'name': names[key]})
        elif 'exponential noise constant interval' == key:
            processes.append({'generator': BasicRandomWalk(diffusion, delta_t, ExponGenerator(), steps, const_diffusion),
                              'name': names[key]})
        elif 'levy noise constant interval' == key:
            processes.append({'generator': BasicRandomWalk(diffusion, delta_t, LevyGenerator(), steps, const_diffusion),
                              'name': names[key]})
        elif 'fat tail powerlaw noise constant interval' == key:
            processes.append({'generator': BasicRandomWalk(diffusion, delta_t, PowerLawGenerator(levy_alpha, delta_t), steps, const_diffusion),
                              'name': names[key]})
        elif 'FBM negative correlation constant interval' == key:
            processes.append({'generator': FBM(diffusion, delta_t, hurst_neg, steps, const_diffusion),
                              'name': names[key]})
        elif 'FBM positive correlation constant interval' == key:
            processes.append({'generator': FBM(diffusion, delta_t, hurst_pos, steps, const_diffusion),
                              'name': names[key]})
        elif 'FBM variable hurst exponent' == key:
            processes.append({'generator': FBM(diffusion, delta_t, hurst_pos, steps, const_diffusion, const_hurst=False),
                              'name': names[key]})
        elif 'CTRW normal noise uniform waiting times' == key:
            processes.append({'generator': CTRW(diffusion, delta_t, NormalGenerator(), uni_time_gen, steps, const_diffusion),
                              'name': names[key]})
        elif 'CTRW symmetric exponential uniform waiting times' == key:
            processes.append({'generator': CTRW(diffusion, delta_t, SymmetricExponGenerator(), uni_time_gen, steps, const_diffusion),
                              'name': names[key]})
        elif 'CTRW exponential noise uniform waiting times' == key:
            processes.append({'generator': CTRW(diffusion, delta_t, ExponGenerator(), uni_time_gen, steps, const_diffusion),
                              'name': names[key]})
        elif 'CTRW levy noise uniform waiting times' == key:
            processes.append({'generator': CTRW(diffusion, delta_t, LevyGenerator(), uni_time_gen, steps, const_diffusion),
                              'name': names[key]})
        elif 'CTRW normal noise constant alpha' == key:
            processes.append({'generator': CTRW(diffusion, delta_t,  NormalGenerator(), PowerLawGenerator(mean_alpha, delta_t), steps,
                                                const_diffusion),
                              'name': names[key]})    
        elif 'CTRW normal noise variable alpha' == key:
            processes.append({'generator': CTRW(diffusion, delta_t,  NormalGenerator(), PowerLawGenerator(alpha, delta_t), steps,
                                                const_diffusion, alpha),
                              'name': names[key]})    
        elif 'CTRW symmetric noise variable alpha' == key:
            processes.append({'generator': CTRW(diffusion, delta_t,  SymmetricExponGenerator(), PowerLawGenerator(alpha, delta_t), steps,
                                                const_diffusion, alpha),
                              'name': names[key]})    
        elif 'CTRW exponential noise variable alpha' == key:
            processes.append({'generator': CTRW(diffusion, delta_t, ExponGenerator(), PowerLawGenerator(alpha, delta_t), steps,
                                                const_diffusion, alpha),
                              'name': names[key]})    
        elif 'CTRW levy noise variable alpha' == key:
            processes.append({'generator': CTRW(diffusion, delta_t, LevyGenerator(), PowerLawGenerator(alpha, delta_t), steps,
                                                const_diffusion, alpha),
                              'name': names[key]})    
    
    return processes


def save_dataset(train_set, test_set, train_name, test_name, parameters, param_file):
    # save the train and test sets as csv    
    train_set.to_csv(train_name)
    test_set.to_csv(test_name)
    
    with open(param_file, 'w') as fp:
        json.dump(parameters, fp, sort_keys=True, indent=True)
    
    
def create_dataset(diffusion, delta_t, steps, train_walkers, test_walkers, names, train_name, test_name,
                   hurst_neg=0.3, hurst_pos=0.7, param_name='parameters.json', alpha=3, preprocess=[]):
    # decide whether all the walks have the same diffusion constant or a diffusion constant taken from some distribution
    const_diffusion = type(diffusion) == float
    # decide the length of the walk, the interval between each measurement and the amount of samples from each walk
    columns = [f'step {i+1}' for i in range(steps)]
    columns.append('diffusion')
    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    classes = list(set(names.values()))
    val_dict = {'names': classes, 'length': steps, 'delta_t': delta_t, 'diffusion const': diffusion if const_diffusion else 'variable'}
    
    # create a list of processes and names
    processes = createprocesses(diffusion=diffusion, delta_t=delta_t, steps=steps, names=names, const_diffusion=const_diffusion,
                                hurst_neg=hurst_neg, hurst_pos=hurst_pos, alpha=alpha)
    
    train_walkers, test_walkers = int(train_walkers / len(classes)), int(test_walkers / len(classes))

    for process in processes:
        # generating a random walk according to the current process
        walks = pd.DataFrame(process['generator'].generate_walks(train_walkers + test_walkers), columns=columns)
        # add a new column with the process name as value
        walks['model'] = process['name']
        # splitting into train and test sets and appending to the total set
        train_set = train_set.append(walks.iloc[:train_walkers, :], ignore_index=True)
        test_set = test_set.append(walks.iloc[train_walkers:, :], ignore_index=True)
    
    # # scale the data this is not used as the random walks should 
    # train_set, scaler = scale_data(train_set)
    # test_set, _ = scale_data(test_set, scaler)
    
    # # this part is responsible for preprocessing the datas
    # for pre in preprocess:
    #     train_set = globals()[pre](train_set)
    #     test_set = globals()[pre](test_set)
        
    save_dataset(train_set, test_set, train_name, test_name, val_dict, param_name)
    return train_set, test_set


def generate_mixed_walk(processes, steps, diffusion, transition_range, start_point=None):
    temp_processes = processes.copy()
    # choose two different processes
    process1 = random.choice(temp_processes)
    temp_processes.remove(process1)
    process2 = random.choice(temp_processes)
    # make sure the model is different even for cases like CTRW with different noise functions
    while process2['name'] == process1['name']:
        temp_processes.remove(process2)
        process2 = random.choice(temp_processes)
    # choose a random transition point in the given range
    transition = random.randint(transition_range[0], transition_range[1])
    # generating a random walk according to the current process
    start = start_point.generate_variate() if start_point != None else 0
    # walk1 = process1['generator'].generate_walks(amount=1, steps=transition, diffusion=d)
    d1, d2 = diffusion.generate_variate(), diffusion.generate_variate()
    walk1 = process1['generator'].generate_walks(amount=1, steps=transition, diffusion=d1, start_point=start)
    walk2 = process2['generator'].generate_walks(amount=1, start_point=walk1[0, -2], steps=steps-transition, diffusion=d2)
    # walk2 = process2['generator'].generate_walks(amount=1, start_point=walk1[0, -2], steps=steps-transition, diffusion=d1)
    return np.append(walk1[0, :-1], walk2[0, :]).reshape(1, steps+1), transition, (process1['name'], process2['name']) 


def rearange_dataset_columns(df, classification_columns):
    """
    creates a list with ordered columns where the steps are at the beginning and the classifications at the end
    """
    return df[[col for col in df if col not in classification_columns] + classification_columns]
            

def createmixdataset(diffusion, delta_t, steps, train_walkers, test_walkers, names, train_name, test_name,
                      hurst_neg=0.3, hurst_pos=0.7, max_transitions=2, transition_prob=1e-2,
                      param_name='parameters.json', alpha=3, preprocess=[]):
    """
    a method to create a dataset of random walks with a random transition point. the last column (label) is the step where the 
    transition occured.
    """
    # decide whether all the walks have the same diffusion constant or a diffusion constant taken from some distribution
    const_diffusion = type(diffusion) == float
    # decide the length of the walk, the interval between each measurement and the amount of samples from each walk
    columns = [f'step {i+1}' for i in range(steps)]
    classes = list(set(names.values()))
    columns.append('diffusion')
    transition_range = [int(steps/4), int(3*steps/4)]
    labels = list(itertools.permutations(classes, 2))
    # create a dictionary for all the information about the dataset
    val_dict = {'names': labels, 'length': steps, 'delta_t': delta_t, 'diffusion const': diffusion if const_diffusion else 'variable'}
    
    # create a list of processes and names
    processes = createprocesses(diffusion=diffusion, delta_t=delta_t, steps=steps, names=names, const_diffusion=const_diffusion,
                                hurst_neg=hurst_neg, hurst_pos=hurst_pos, alpha=alpha)
    
    sample_amount = train_walkers + test_walkers
    tot_walks = pd.DataFrame()
    for i in range(sample_amount):
        # generating a new diffusion constant
        d = diffusion.generate_variate()
        # generating a random walk according to the current process
        # check whether or not a transition should occur
        transition_happened = random.random() < transition_prob
        if transition_happened:
            walk, transition, proc_names = generate_mixed_walk(processes, steps, d, transition_range)
        else:
            # choose a random process and generate a walk from it
            process = random.choice(processes)
            walk = process['generator'].generate_walks(amount=1, steps=steps, diffusion=d)
            proc_names = (process['name'], process['name'])
            transition = 0
        walk = pd.DataFrame(walk, columns=columns)
        # add columns for different classification options
        walk['model'] = str(list(proc_names))
        walk['mixed'] = transition_happened
        walk['transition location'] = transition
        # append the walk to the total dataset
        tot_walks = tot_walks.append(walk, ignore_index=True)
    # splitting into train and test sets and appending to the total set
    train_set = tot_walks.iloc[:train_walkers, :]
    test_set = tot_walks.iloc[train_walkers:, :]
    # add the non-mixed models to the classification options
    val_dict['names'].extend([(clas, clas) for clas in classes])
    
    # # scale the data
    # train_set, scaler = scale_data(train_set)
    # test_set = scale_data(test_set, scaler)
    
    # for pre in preprocess:
    #     train_set = globals()[pre](train_set)
    #     test_set = globals()[pre](test_set)
 
    save_dataset(train_set, test_set, train_name, test_name, val_dict, param_name)
    return train_set, test_set


def create_variable_length_dataset(diffusion, delta_t, steps, train_walkers, test_walkers, names, train_name, test_name,
                      hurst_neg=0.3, hurst_pos=0.7, max_transitions=2, transition_prob=1e-2,
                      param_name='parameters.json', alpha=3, preprocess=[], min_steps=25, start_point=None, levy_alpha=1.75, 
                      tran_range=[1/4, 3/4]):
    """
    a method to create a dataset of random walks with a random transition point. the random walks have variable lengths (0.25*steps, steps)
    """
    # decide whether all the walks have the same diffusion constant or a diffusion constant taken from some distribution
    const_diffusion = type(diffusion) == float
    # decide the length of the walk, the interval between each measurement and the amount of samples from each walk
    classes = list(set(names.values()))
    transition_range = [int(tran_range[0] * steps), int(tran_range[1] * steps)]
    # check if there are transitions required
    labels = list(itertools.permutations(classes, 2)) if transition_prob != 0 else classes
    transition_enabled = transition_prob != 0
    # create a dictionary for all the information about the dataset
    val_dict = {'names': labels, 'length': steps, 'delta_t': delta_t, 'diffusion const': diffusion if const_diffusion else 'variable'}
    # add the non-mixed models to the classification options if transition probability is smaller than 1
    if transition_prob < 1 and transition_prob > 0:
        val_dict['names'].extend([(clas, clas) for clas in classes])
    
    # create a list of processes and names
    processes = createprocesses(diffusion=diffusion, delta_t=delta_t, steps=steps, names=names, const_diffusion=const_diffusion,
                                hurst_neg=hurst_neg, hurst_pos=hurst_pos, alpha=alpha, levy_alpha=levy_alpha)
    
    sample_amount = train_walkers + test_walkers
    tot_walks = pd.DataFrame()
    for i in range(sample_amount):
        # generating a new diffusion constant
        d = diffusion.generate_variate()
        walk_length = random.randint(min_steps, steps)
        columns = [f'step {i+1}' for i in range(walk_length)] + ['diffusion']
        # check whether or not a transition should occur (also depends if there should be transitions in the datset)
        transition_happened = 0 < random.random() < transition_prob
        if transition_happened:
            walk, transition, proc_names = generate_mixed_walk(processes, walk_length, diffusion, transition_range)
        else:
            # choose a random process and generate a walk from it
            process = random.choice(processes)
            # generate a random starting point for the walk
            start = start_point.generate_variate() if start_point != None else 0
            walk = process['generator'].generate_walks(amount=1, steps=walk_length, diffusion=d, start_point=start)
            # walk = process['generator'].generate_walks(amount=1, steps=walk_length, diffusion=d)
            proc_names = (process['name'], process['name'])
            transition = 0
        
        # #normalize the walk
        # walk[:, :-1] = preprocessing.normalize(walk[0, :-1], norm='l1')
        
        walk = pd.DataFrame(walk, columns=columns)
        # walk = pd.DataFrame(walk)
        # add columns for different classification options
        walk['model'] = str(tuple(proc_names)) if transition_enabled else proc_names[0]
        walk['mixed'] = transition_happened
        walk['transition location'] = transition
        walk['length'] = walk_length
        # append the walk to the total dataset
        tot_walks = pd.concat([tot_walks, walk], ignore_index=True, sort=False)
    
    label_names = ['diffusion', 'model', 'mixed', 'transition location', 'length']
    tot_walks = rearange_dataset_columns(tot_walks, label_names)
    # splitting into train and test sets and appending to the total set
    train_set = tot_walks.iloc[:train_walkers, :]
    test_set = tot_walks.iloc[train_walkers:, :]
    
    # # scale the data
    # train_set, scaler = scale_data(train_set, label_names)
    # test_set, _ = scale_data(test_set, label_names, scaler)
    
    # for pre in preprocess:
    #     train_set = globals()[pre](train_set)
    #     test_set = globals()[pre](test_set)
 
    save_dataset(train_set, test_set, train_name, test_name, val_dict, param_name)
    return train_set, test_set


if __name__ == '__main__':
    names = ['Normal noise constant interval', 'symmetric exponential noise constant interval', 'exponential noise constant interval',
 'FBM negative correlation constant interval', 'FBM positive correlation constant interval',
'CTRW normal noise uniform waiting times', 'CTRW symmetric exponential uniform waiting times',
'CTRW exponential noise uniform waiting times']

    # create_dataset(diffusion=0.4, delta_t=0.05, steps=100, train_walkers=200, test_walkers=100, names=names,
    #                train_name='train_set.csv', test_name='test_set.csv')
    createmixdataset(diffusion=UniformGenerator(0,1), delta_t=0.05, steps=50, train_walkers=2000, test_walkers=100, names=names,
                   train_name='train_set.csv', test_name='test_set.csv')
    
