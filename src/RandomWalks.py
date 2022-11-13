# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:27:57 2020

@author: gedadav
"""
import numpy as np
from Generator import NormalGenerator, UniformGenerator, ExponGenerator, LevyGenerator
from stochastic.processes.continuous import FractionalBrownianMotion, BrownianMotion
import matplotlib.pyplot as plt

""" calculates the mean squared displacement of some particle given a set of random walks
return:
    a numpy array of the mean squared displacement for every time step of the set of walks
"""
def MSD(walks):
    for column in range(walks.shape[1]):
        walks[:, column] -= walks[:, 0]
    sq_disp = np.square(walks)
    mean_sq_disp = np.mean(sq_disp, axis=0)
    return mean_sq_disp


def TA_MSD(walks):  
    """
    calculates the time averaged mean squared displacement of some 1D random walk given as a vector walk
    the Time Averaged Mean Squared Displacement is calculated according to the following equation (Latex form)
    TA-MSD_{T,\Delta}(n) = \frac{1}{T-n\Delta} \sum_{t=0}^{T-n\Delta} (x_{t+n\Delta} - x_t)^2
    """
    delta_t = 1
    T = walks.shape[1]
    # keeping a matrix of squared values and results for quicker computation
    x_squared = np.square(walks)
    tamsd = np.zeros(walks.shape)
    for n in range(0, T, delta_t):
        cur_msd = np.zeros((walks.shape[0], T-n*delta_t))
        for t in range(T - n*delta_t):
            cur_msd[:, t] = x_squared[:, t + n*delta_t] + x_squared[:, t] - 2 * walks[:, t + n*delta_t] * walks[:, t]
        tamsd[:, n] = np.mean(np.array(cur_msd), axis=1)
    return tamsd


def ensemble_TA_MSD(walks, ensemble=100):
    # getting a ta-msd ensemble with 0.1 of the total amount of walks
    ensemble_tamsd = TA_MSD(walks[:ensemble, :])
    # getting the mean of the msd ensemble
    return np.mean(ensemble_tamsd, axis=0)


# takes 2 random walks and measures covariance. this is used as a a benchmark against numpy's covariance
def covariance(x, y):
    mu_x, mu_y = np.mean(x), np.mean(y)
    new_vec = (x - mu_x) * (y - mu_y)
    cov = np.mean(new_vec)
    return cov


# takes 2 random walks and measures correlation. this is used as a benchmark against numpy's correlation
def correlation(x, y):
    cov = covariance(x, y)
    var_x, var_y = np.var(x), np.var(y)
    corr = cov / np.sqrt(var_x * var_y)
    return corr


# takes 2 random walks and measures correlation. this is used as a benchmark against numpy's correlation
def conv_correlation(x, y):
    corr = np.zeros(x.shape)
    corr[0] = np.dot(y, x) / len(x)
    for lag in range(1, len(x)):
        # shift the entire time series lag steps by using numpy's pad method
        lag_x = x.copy()
        lag_x[-lag:] = 0
        corr[lag] = np.dot(y, lag_x) / (len(x) - lag)
    return corr


def correlate(walk1, walk2):
    # diffwalk1 , diffwalk2 = my_diff(walk1), my_diff(walk2)
    diffwalk1 , diffwalk2 = np.diff(walk1), np.diff(walk2)
    corr = 0
    for i in range(len(diffwalk1)):
        for j in range(len(diffwalk2)):
            corr += diffwalk1[i] * diffwalk2[j]
    return corr


# returns the mean autocorrelation as a function of lag time
def auto_corr_walks(walks):
    amount, steps = walks.shape[0], walks.shape[1]
    correlations = np.zeros(walks.shape)
    for lag in range(1, steps):
        for i in range(amount):
            correlations[i, lag] = correlate(walks[i, :], walks[i, :-lag])
    correlations[:, 0] = np.array([correlate(walks[i,:], walks[i,:]) for i in range(amount)])
    return np.fliplr(correlations)


# # returns the mean autocorrelation as a function of lag time
# def ensemble_auto_corr_walks(walks, steps, amount):
#     mean_corr = np.zeros(steps)
#     for lag in range(1, steps):
#         for i in range(amount):
#             mean_corr[lag] += correlate(walks[i, :], walks[i, :-lag])
#         mean_corr[lag] = mean_corr[lag] / amount
#     mean_corr[0] = np.mean([correlate(walks[i,:], walks[i,:]) for i in range(amount)])
#     return np.flip(mean_corr)


# returns the mean autocorrelation as a function of lag time
def ensemble_auto_corr_walks(walks, steps, amount):
    """
    returns the mean of the autocorrelation over an ensemble of random walks as a function of th lag time

    Parameters
    ----------
    walks : np.array
        DESCRIPTION.
    steps : int
        DESCRIPTION.
    amount : int
        DESCRIPTION.
    """
    return np.mean(auto_corr_walks(walks), axis=0)

        
class BasicRandomWalk:
    """
    creates a random walk given a certain diffusion constant, delta and a noise generator.
    """
    def __init__(self, diffusion_generator, delta, generator=NormalGenerator(), steps=100, const_diff=False):
        self.d = diffusion_generator
        self.delta = delta
        self.generator = generator
        self.steps = steps
        self.const_diff = const_diff
        self.walks = np.empty((0, steps + 1))
    
    def generate_walks(self, amount=1, start_point=0, steps=None, diffusion=0):
        if steps is not None:
            self.steps = steps
        for i in range(amount):
            # this part is defined so that for multiple transition walks we will not have different diffusion constants
            if diffusion == 0:
                d = self.d if self.const_diff else self.d.generate_variate()
            else:
                d = diffusion            
            noise = np.sqrt(d * self.delta) * self.generator.generate_tensor(shape=self.steps-1, torch_tensor=False)
            # noise = np.sqrt(d) * self.generator.generate_tensor(shape=self.steps-1, torch_tensor=False)
            # add the starting point to the noise at t=0
            noise = np.insert(noise, 0, start_point)
            # add the diffusion constant at the end of the walk and sum all variables 
            walk = np.append(np.cumsum(noise), d).reshape((1, self.steps+1))
            if steps is not None:
                return walk
            self.walks = np.append(self.walks, walk, axis=0)
        return self.walks
    
    def change_generator(self, new_generator):
        self.generator = new_generator
        
    def change_walks_class(self, new_class):
        self.walks[:][-1] = new_class
        
    def get_walks(self):
        return self.walks
    
        

class CTRW:
    """
    creates a random walk with step sizes and waiting times taken from given distributions
    """
    def __init__(self, diffusion_generator, delta, noise_generator=NormalGenerator(), time_generator=UniformGenerator(), steps=100,
                 const_diff=False, variable_alpha=None):
        self.d = diffusion_generator
        self.delta = delta
        self.noise_gen = noise_generator
        self.time_gen = time_generator
        self.variable_alpha = variable_alpha
        self.steps = steps
        self.const_diff = type(diffusion_generator) == float
        self.walks = np.empty((0, steps + 1))
        
    def generate_walks(self, amount=1, start_point=0, steps=None, diffusion=0):
        if steps is not None:
            self.steps = steps
        # if we chose powerlaw time with variable alpha within the given range
        if self.variable_alpha is not None:
            alpha = UniformGenerator(self.variable_alpha[0], self.variable_alpha[1])
        for i in range(amount):
            if diffusion == 0:
                d = self.d if self.const_diff else self.d.generate_variate()
            else:
                d = diffusion  
            if self.variable_alpha is not None:
                self.time_gen.update(alpha.generate_variate())
            tau = self.generate_waiting_times()
            noise = np.sqrt(d) * self.generate_walk(tau)
            # add the starting point to the noise at t=0
            noise[0] += start_point
            # add the diffusion constant at the end of the walk and sum all variables 
            walk = np.append(np.cumsum(noise), d).reshape((1, self.steps+1))
            # add the current walk to the end of the existing list of walks
            if steps is not None:
                return walk
            self.walks = np.append(self.walks, walk, axis=0)
        return self.walks
    
    def generate_waiting_times(self):
        # create an empty array for the tau's to go into
        time_arr = []
        total_time = 0
        # keep generating waiting times until the total time equals the N*delta_t
        while total_time < (self.steps * self.delta):
            rand_time = self.time_gen.generate_variate()
            time_arr.append(rand_time)
            total_time += rand_time
        return np.asarray(time_arr)
    
    def generate_walk(self, tau):
        # create a random walk and fill it with N steps according to the random walk
        walk = []
        # create a sum array of all the waiting times
        taus = iter(np.cumsum(tau))
        cur_tau = next(taus)
        # timer to check how long it has been since the start of the experiment
        timer = 0
        while len(walk) < self.steps:
            # if the current time is still lower than the next waiting time for jump = if it is not time to jump, do nothing
            if timer <= cur_tau:
                walk.append(0)
            else:
                # take the next step and advance to look at the next jump time
                walk.append(self.noise_gen.generate_variate())
                cur_tau = next(taus)
            # take the next time step of size delta_t = next measurement of the system
            timer += self.delta
        return np.asarray(walk)                
    
    def change_noise_generator(self, new_generator):
        self.noise_gen = new_generator
        
    def change_time_generator(self, new_generator):
        self.time_gen = new_generator
        
    def change_walks_class(self, new_class):
        self.walks[:][-1] = new_class
        
    def get_walks(self):
        return self.walks
    
    
class FBM:
    """
    creates a Fractional Brownian Motion random walk
    """
    # make it work for more than 1D
    def __init__(self, diffusion_generator, delta, hurst, steps=100, const_diff=False, const_hurst=True):
        self.d = diffusion_generator
        self.delta = delta
        self.h = hurst
        self.generator = FractionalBrownianMotion(hurst=hurst, t=steps*delta)
        self.steps = steps
        self.delta = delta
        self.const_hurst = const_hurst
        # bring this back to get regular FBM
        self.hurst_gen = UniformGenerator(0, 1)
        # add this to get Brownian Motion
        # self.hurst_gen = UniformGenerator(0.49, 0.51)
        self.const_diff = type(diffusion_generator) == float
        self.walks = np.empty((0, steps + 1))
    
    def generate_walks(self, amount=1, start_point=0, steps=None, diffusion=0):
        if steps is not None:
            self.steps = steps
        for i in range(amount):
            if not self.const_hurst:
                # keep generating hurst values until you have one that is not 0.5 (+- 0.1)
                hurst = self.hurst_gen.generate_variate()
                # keep this to make FBM without brownian motions
                while (hurst > 0.4 and hurst < 0.6):
                    hurst = self.hurst_gen.generate_variate()
                self.generator = FractionalBrownianMotion(hurst=hurst, t=self.steps*self.delta)
            # no need to reset the generator
            if diffusion == 0:
                d = self.d if self.const_diff else self.d.generate_variate()
            else:
                d = diffusion
            walk = np.sqrt(d) * (self.generator.sample(self.steps - 1))
            # add the starting point to the noise at t=0
            walk += start_point
            # add the diffusion constant at the end of the walk and sum all variables 
            walk = np.append(walk, d).reshape((1, self.steps+1))
            if steps is not None:
                return walk
            self.walks = np.append(self.walks, walk, axis=0)
        return self.walks
        
    def change_walks_class(self, new_class):
        self.walks[:][-1] = new_class
        
    def get_walks(self):
        return self.walks
    
    def change_noise_generator(self, new_generator):
        self.noise_gen = new_generator
        
    
    