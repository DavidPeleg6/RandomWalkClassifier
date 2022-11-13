# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:45:47 2020

@author: gedadav
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_and_unpack_batch(batch):
    """
    checks whether there is a nan value indicating that were using a variable length model so in will convert x into a packed sequence
    """
    # if there are no variable length walks, return original batch
    walks = torch.isnan(batch)
    if not (True in walks):
        return batch
    lengths = []
    # find the size of all walks in the batch and add it to lengths
    for x in walks:
        # find the index of the first nan in the array
        walk_len = (x == True).nonzero(as_tuple=True)
        # if we got the maximal walk length
        if len(walk_len[0]) < 1:
            lengths.append(int(batch.shape[1]) - 1)
        else:
            lengths.append(int(walk_len[0][0]) - 1)
    batch = batch.view(batch.shape[0], batch.shape[1], 1)
    return torch.nn.utils.rnn.pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=False)


def truncate_walk(walk):
    # if there are no variable length walks, return original batch
    nans = torch.isnan(walk)
    if not (True in nans):
        return walk
    walk_len = (nans == True).nonzero(as_tuple=True)
    return walk[0][:(int(walk_len[1][0]))].reshape((1, int(walk_len[1][0])))


def train(train_loader, model, optimizer):
    """
    return:
        mean loss of the train
        accuracy of the train
    """   
    model.train()
    running_loss = 0
    correct_train = 0
    batch_size = train_loader.batch_size
    # model = model.to(device)
    # according to the data loader, the data is sent in mini batches of 10
    for data in train_loader:
        # taking a set of features and labels
        X, y = data[0].to(device), data[1].to(device)
        # these functions were used for packing the sequences in order to feed into RNN in batch
        # X = truncate_walk(X)
        # X = check_and_unpack_batch(X)
        # we have to reset the gradient after each iteration
        optimizer.zero_grad()
        output = model(X)
        # a scalar value, we will use negative log likelyhood
        # if our data is a one-hot vector we use mean squared error
        loss = F.nll_loss(output, y)
        # we would now like to backpropogate the loss and compute the gradients
        loss.backward()
        # for now we can only use the manual learning without fancy adam
        optimizer.step()
        # update our mean loss and accuracy counters
        running_loss += loss.item()
        # get the predicted values
        predicted = output.max(1, keepdim=True)[1]
        # get the sum of all the wrong label predictions in current batch
        correct_train += int(predicted.eq(y.view_as(predicted)).cpu().sum())
    # return the mean loss and mean accuracy
    return running_loss/(len(train_loader)*int(batch_size)), 100 * (correct_train / (len(train_loader)*int(batch_size)))
            

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
            X, y = X.to(device), y.to(device)
            # X = check_and_unpack_batch(X)
            # X = truncate_walk(X)
            output = model(X)
            v_loss += F.nll_loss(output, y).item()
            pred = output.max(1, keepdim=True)[1]
            correct += int(pred.eq(y.view_as(pred)).cpu().sum())
    return v_loss/(len(validation_loader)*int(batch_size)), 100 * (correct / (len(validation_loader)*int(batch_size)))


def regression_train(train_loader, model, optimizer):
    """
return:
    mean loss of the train
    accuracy of the train
"""   
    model.train()
    running_loss = 0
    # mae loss
    loss_func = nn.SmoothL1Loss()
    # according to the data loader, the data is sent in mini batches of 10
    for data in train_loader:
        # taking a set of features and labels
        X, y = data[0].to(device), data[1].to(device).to(torch.float32)
        # we have to reset the gradient after each iteration
        optimizer.zero_grad()
        output = model(X)
        # a scalar value, we will use negative log likelyhood
        # if our data is a one-hot vector we use mean squared error
        loss = loss_func(output.view(-1), y)
        # we would now like to backpropogate the loss and compute the gradients
        loss.backward()
        # for now we can only use the manual learning without fancy adam
        optimizer.step()
        # update our mean loss and accuracy counters
        running_loss += loss.item()
    # return the mean loss and mean accuracy
    return 0, running_loss/len(train_loader)
            

def regression_val(validation_loader, model):
    """
returns:
    the validation set mean loss
    the validation set mean accuracy
"""
    model.eval()
    # mae loss
    loss_func = nn.SmoothL1Loss()
    with torch.no_grad():
        v_loss = 0
        for data in validation_loader:
            # split into label and sample and run them through the network
            X, y = data
            X, y = X.to(device), y.to(device)
            output = model(X).view(-1)
            y = y.to(torch.float32)
            output, y = torch.round(output), torch.round(y)
            v_loss += loss_func(output, y).item()
    return 0, v_loss/len(validation_loader)


"""
this method is used mainly for validation in the transition regression problem
returns:
    the accuracy of the model on the validation set
"""
def special_val(validation_loader, model):
    model.eval()
    batch_size = validation_loader.batch_size
    correct = 0
    with torch.no_grad():
        correct = 0
        for data in validation_loader:
            # split into label and sample and run them through the network
            X, y = data
            X, y = X.to(device), y.to(device)
            output = model(X).view(-1)
            y = y.to(torch.float32)
            output, y = torch.round(output), torch.round(y)
            for i in range(output.shape[0]):
                correct += 1 if output[i] == y[i] else 0
    return 0, 100 * (correct / (len(validation_loader)*int(batch_size)))


def validate_mixed(X, model, split_amount):
    # making the model classify half of the random walk given
    # check if transition happened in the first half
    # half, quarter = int(X.shape[1] / 2), int(X.shape[1] / 4)
    # x1, x2, x3, x4 = X[0, :quarter], X[0, :half], X[0, :half+quarter], X[0, :]
    # # first case = transition happens in first quarter
    # if model(x1.view(1, x1.shape[0])).max(1, keepdim=True)[1] != model(x2.view(1, x2.shape[0])).max(1, keepdim=True)[1]:
    #     X1, X2 = x1.view(1, x1.shape[0]), X[0, quarter:X.shape[1]].view(1, half + quarter)
    #     output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    # # second case = transition happens in the middle
    # elif model(x2.view(1, x2.shape[0])).max(1, keepdim=True)[1] != model(x3.view(1, x3.shape[0])).max(1, keepdim=True)[1]:
    #     X1, X2 = X[0, :half].view(1, half), X[0, half:].view(1, half)
    #     output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    # # third case = transition happens in the third quarter
    # elif model(x3.view(1, x3.shape[0])).max(1, keepdim=True)[1] != model(x4.view(1, x4.shape[0])).max(1, keepdim=True)[1]:
    #     X1, X2 = X[0, 0:half+quarter].view(1, half+quarter), X[0, half+quarter:].view(1, quarter)
    #     output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    # else:
    #     output = (model(X).max(1, keepdim=True)[1], model(X).max(1, keepdim=True)[1]) 
    
    
    # # second method
    # split_index = int(X.shape[1] / 2)
    # X1, X2 = X[0, :split_index].view(1, split_index), X[0, split_index:].view(1, split_index)
    # output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    
    # # third method
    # split_index = int(X.shape[1] / 4)
    # X1, X2 = X[0, :split_index].view(1, split_index), X[0, 3 * split_index:].view(1, split_index)
    # output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    
    # # fourth method
    # split_index = int(X.shape[1] / 2)
    # X1, X2 = X[0, :split_index].view(1, split_index), X
    # output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    
    # # fifth method - subtract max val from the entire walk
    # split_index = int(X.shape[1] / 4)
    # X1, X2 = X[0, :split_index].view(1, split_index), X[0, 3*split_index:].view(1, split_index)
    # # take the maximal\minimal value of the unexplored range of the walk
    # max_val = torch.max(X[0, split_index:3*split_index]) if torch.max(X2) > 0 else torch.min(X[0, split_index:3*split_index])
    # # subtract the maximal value from the walk to keep it close to 0
    # X2 = torch.absolute(X2 - max_val)
    # output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    
    # sixth method - make walks start from 0
    split_index = int(X.shape[1] / split_amount)
    X1, X2 = X[0, :split_index].view(1, split_index), X[0, (split_amount - 1)*split_index:].view(1, split_index)
    # if there was a levy flight in the segment (meaning a value larger than 500)
    subtract_val = X[0, (split_amount - 1)*split_index]
    # subtract the maximal value from the walk to keep it close to 0
    X2 = torch.absolute(X2 - subtract_val)
    output = (model(X1).max(1, keepdim=True)[1], model(X2).max(1, keepdim=True)[1])
    
    # # seventh method - sixth method but with variable transition amount
    # split_index = int(X.shape[1] / split_amount)
    # output = []
    # for i in range(1, split_amount):
    #     X1 = X[0, (i-1)*split_index:i*split_index].view(1, split_index)
    #     subtract_val = X[0, (i-1)*split_index]
    #     # subtract the maximal value from the walk to keep it close to 0
    #     X1 = torch.absolute(X1 - subtract_val)
    #     out = model(X1).max(1, keepdim=True)[1] 
    #     if out not in output:
    #         output.append(out)
    
    # if len(output) == 1:
    #     output.append(output[0])
    
    return output

def mixed_model_validator(loader, model, splits=1, custom_validator=True, split_index=4):
    """
    this method classifies transitions made in the code in the following manner: 
        1. The walk is split into segments (starting with 1 but as we get longer walks we might add more. for now its ignored)
        2. Every segment is given to the network for classification
    Note that the batch sized is ignored in this case as the packed rnn swaps values with 0 which ruins classification
    
    Parameters
    ----------
    loader : DataLoader
        the dataloader for the dataset. will only work in batches of 1
    model : torch.nn
        a RNN model trained for classifying a single random walk
    splits : int
        the amount of splits the trajectory can contain at most
    split_index : int
        to how many parts the trajectory should be split into        

    Returns
    -------
    accuracy : int
        the accuracy of the model on the given dataset

    """
    model.eval()
    correct = 0
    if loader.batch_size > 1:
        raise ValueError('the batch size should be 1')
    with torch.no_grad():
        for data in loader:
            X, y = data
            X, y = X.to(device), y.to(device)
            y = (y[0][0], y[0][1])
            if custom_validator:
                output = validate_mixed(X, model, split_index)
            else:
                output = model(X).max(1, keepdim=True)[1]
            if output[0] == y[0] and output[1] == y[1]:
                correct += 1
    return 100 * (correct / len(loader))
            
            
        
    




