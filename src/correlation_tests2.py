# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:57:21 2021

@author: gedadav
"""

import Generator
from RandomWalks import BasicRandomWalk, CTRW, FBM
from RandomWalks import TA_MSD, MSD
import matplotlib.pyplot as plt
from Generator import NormalGenerator, UniformGenerator, ExponGenerator, LevyGenerator, PowerLawGenerator, SymmetricExponGenerator, FractionalGaussianNoise
from stochastic.processes.continuous import FractionalBrownianMotion, BrownianMotion
import seaborn as sns
import pandas as pd
import time
from scipy.special import gamma
import random
import numpy as np

# Use seaborn style defaults and set the default figure size
sns.set()



# creating a constant diffusion Brownian motion to test
const_diffusion = True
# diffusion = 0.3
# delta_t = 0.05
diffusion = 1
delta_t = 1
steps = 100
amount = int(1e2)
times = np.linspace(0, steps*delta_t, steps)

mu = 0
sigma = 1

"""
-------------------------------------------------------------------------------
                        generating everything from scratch!!!!!!!
-------------------------------------------------------------------------------
"""
# def my_diff(walk):
#     diff = walk.copy()[1:]
#     for i in range(1, len(walk)):
#         diff[i-1] = walk[i] - walk[i-1]
#     return diff


def correlate(walk1, walk2):
    # diffwalk1 , diffwalk2 = my_diff(walk1), my_diff(walk2)
    diffwalk1 , diffwalk2 = np.diff(walk1), np.diff(walk2)
    corr = 0
    for i in range(len(diffwalk1)):
        for j in range(len(diffwalk2)):
            corr += diffwalk1[i] * diffwalk2[j]
    return corr


def ensemble_cross_correlate(walks, t):
    choices = amount * 30
    mean_corr = 0
    for i in range(choices):
        choice1 = random.randint(0, walks.shape[0]-1)
        choice2 = random.randint(0, walks.shape[0]-1)
        while choice2 == choice1: 
            choice2 = random.randint(0, walks.shape[0]-1)
        walk1, walk2 = walks[choice1, :], walks[choice2, :]
        mean_corr += correlate(walk1[:t], walk2[:t])
    return mean_corr / choices


# returns the mean autocorrelation as a function of lag time
def auto_corr_walks(walks, steps, amount):
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
    return np.mean(auto_corr_walks(walks, steps, amount), axis=0)


def auto_cov_walks(walks):
    mean_cov = np.zeros(steps)
    # mean_walk1, mean_walk2 = np.zeros(steps), np.zeros(steps)
    mean_walk = np.zeros(steps)
    for lag in range(1, steps):
        for i in range(amount):
            walk1, walk2 = walks[i, :], walks[i, :-lag]
            mean_cov[lag] += correlate(walk1, walk2)
            # mean_walk1[lag] += sum(walk1)
            # mean_walk2[lag] += sum(walk2)
        mean_walk[lag] = np.mean(walks[:, -lag])
        # mean_cov[lag] = (mean_cov[lag] / amount) - ((mean_walk1[lag] / amount) * (mean_walk2[lag] / amount))
        mean_cov[lag] = (mean_cov[lag] / amount) - mean_walk[lag]**2
    mean_cov[0] = np.mean([correlate(walks[i,:], walks[i,:]) for i in range(amount)])
    return np.flip(mean_cov)

    
 
    
# # #---------------------------------Brownian Motion
# # initialize a noise vector
# noise = np.sqrt(diffusion * delta_t) * np.random.normal(mu, sigma, (amount, steps))
# # set the starting point to be 0
# noise[:, 0] = 0
# # sum rows
# walks = np.cumsum(noise, axis=1)
# real_autocorr = diffusion * times


#-------------------------------- FBM
for hurst in np.linspace(0.05, 0.95, 10):
    print('hurst exponent: {}'.format(hurst))
    alpha = hurst * 2
    # anomalous_diff = diffusion / gamma(1+alpha)
    anomalous_diff = diffusion
    mygen = FractionalBrownianMotion(hurst=hurst, t=steps*delta_t)
    walks = np.array([np.sqrt(diffusion * delta_t) * mygen.sample(steps-1) for i in range(amount)])
    # walks = np.array([mygen.sample(steps-1) for i in range(amount)])
    # # creating the FBM auto correlation function
    real_autocorr = np.ones(steps)
    T = steps * delta_t
    # the starting time of the correlation is t and the final time is T=steps*delta_t
    for t in range(steps):
        real_autocorr[t] = 0.5 * anomalous_diff * (times[t]**alpha + T**alpha - np.abs(T - times[t])**alpha)
        # real_autocorr[t] = 0.5 * (times[t]**alpha + T**alpha - np.abs(T - times[t])**alpha)        
    # real_auto = np.flip(real_auto)


# #-------------------------------- Custom FBM
# for hurst in np.linspace(0.05, 0.95, 10):
#     # hurst = 0.8
#     alpha = hurst * 2
#     # diffusion = diffusion / gamma(1+alpha)
#     # anomalous_diff = diffusion / gamma(1+alpha)
#     mygen = FractionalGaussianNoise(steps=steps, hurst=hurst)
#     # walks = np.array([anomalous_diff * mygen.sample(steps-1) for i in range(amount)])
#     # walks = np.array([anomalous_diff * np.cumsum(mygen.temp_walk_generation()) for i in range(amount)])
#     walks = np.array([np.sqrt(diffusion * delta_t) * np.cumsum(mygen.temp_walk_generation()) for i in range(amount)])
#     # # creating the FBM auto correlation function
#     real_autocorr = np.ones(steps)
#     T = steps * delta_t
#     # the starting time of the correlation is t and the final time is T=steps*delta_t
#     for t in range(steps):
#         # real_autocorr[t] = 0.5 * anomalous_diff * (times[t]**alpha + T**alpha - np.abs(T - times[t])**alpha)
#         real_autocorr[t] = 0.5 * diffusion * (times[t]**alpha + T**alpha - np.abs(T - times[t])**alpha)
#     # real_auto = np.flip(real_auto)
    
    
        
    #-------------------------------- get the mean autocorrelation function and compare it to the real autocorr
    plt.figure()
    plt.plot(times, real_autocorr, label='real')
    auto_corr = ensemble_auto_corr_walks(walks, steps, amount)
    plt.plot(times, auto_corr, label='my')
    # auto_cov = np.flip(auto_cov_walks(walks))
    # plt.plot(times, auto_cov, label='my')
    plt.xlabel('lag time')
    plt.ylabel('<X(t) X(t-lag)>')
    plt.legend()
    plt.title('FBM with hurst = {}'.format(hurst))

# #------------------------------ find the correlation between walks in the array = should be 0 
# correlation = ensemble_cross_correlate(walks, steps-1)
# print(correlation)

# #----------------------------- plot all walks
# plt.figure()
# for i in range(amount):
#     plt.plot(times, walks[i])

# #----------------------------- to test that the walks look like diff and then cumsum    
# walk1 = walks[0]
# plt.figure()
# plt.plot(times, walk1, label='walk')
# newwalk = np.cumsum(np.hstack((0, np.diff(walk1))))
# plt.plot(times, newwalk, 'r+', label='walk new')
# plt.legend()

# #------------------------------- histogram of noise and random walks
# plt.figure()
# plt.hist(noise[:, 1:], 30)
# plt.figure()
# plt.hist(walks[:, 1:], 30)
    

    









