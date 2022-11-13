# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:30:56 2021

@author: gedadav
"""

import Generator
from RandomWalks import BasicRandomWalk, CTRW, FBM
from RandomWalks import TA_MSD, MSD
import matplotlib.pyplot as plt
from Generator import NormalGenerator, UniformGenerator, ExponGenerator, LevyGenerator, PowerLawGenerator, SymmetricExponGenerator
from stochastic.processes.continuous import FractionalBrownianMotion, BrownianMotion
import matplotlib.pyplot as plt
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
amount = int(1e3)
plt.figure()


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

# takes a random walk and measures the autocorrelation of it, uses the correlation function/
# this is used as a benchmark against pandas autocorrelation
def autocorrelation(x, lag=1):
    # shift the entire time series lag steps by using numpy's pad method
    # lag_x = np.pad(x, (lag, 0))[:-lag]
    # shift the entire time series lag steps by using numpy's pad method
    lag_x = x.copy()
    # swap the last values of x_t with 0 to simulate x_{t-lag}
    lag_x[-lag:] = 0
    # check the correlation between the series and the lagged series
    auto_corr = correlation(x, lag_x)
    return auto_corr


def auto_corr_function(x):
    # create a auto-corr vector of zeros
    auto_corr = np.ones(x.shape[0])
    # create the autocorrelation as a function of the time lag
    for time_lag in range(1, x.shape[0] - 1):
        auto_corr[time_lag] = autocorrelation(x, time_lag)
    auto_corr[-1] = 0
    return auto_corr


def numpy_auto_corr_func(x):
    # create a auto-corr vector of zeros
    auto_corr = np.ones(x.shape[0])
    # create the autocorrelation as a function of the time lag
    for time_lag in range(1, x.shape[0] - 1):
        # shift the entire time series lag steps by using numpy's pad method
        lag_x = np.pad(x, (time_lag, 0))[:-time_lag]
        auto_corr[time_lag] = np.corrcoef(x, lag_x)[0,1]
    auto_corr[-1] = 0
    return auto_corr
    
def pandas_auto_corr_func(x):
    # create a auto-corr vector of zeros
    auto_corr = np.ones(x.shape[0])
    # create the autocorrelation as a function of the time lag
    for time_lag in range(1, x.shape[0] - 1):
        auto_corr[time_lag] = x.autocorr(lag=time_lag)
    auto_corr[-1] = 0
    return pd.DataFrame(auto_corr)


# takes 2 random walks and measures correlation. this is used as a benchmark against numpy's correlation
def conv_correlation(x, y):
    corr = np.zeros(x.shape)
    # corr[0] = np.dot(y, x) / len(x)
    corr[0] = np.mean(y * x)
    for lag in range(1, len(x)):
        # shift the entire time series lag steps by using numpy's pad method
        lag_x = x.copy()
        # swap the last values of x_t with 0 to simulate x_{t-lag}
        lag_x[-lag:] = 0
        # calculate <x_t * x_{t-lag}>
        corr[lag] = np.dot(y, lag_x) / (len(x) - lag)
    return corr


def mean_correlation(walks):
    size = 10 * walks.shape[0]
    corr = 0
    for i in range(size):
        walk1, walk2 = walks[random.randint(0, walks.shape[0]-1)], walks[random.randint(0, walks.shape[0]-1)]
        corr += np.mean(walk1 * walk2)
    return corr / size

# """
# calculates the mean correlation in the walks dataset, will be used to benchmark my correlation vs numpy
# """
# def numpy_mean_correlation(walks):
#     # initialize an array for the mean correlations as a function of time
#     mean_correlations = np.ones(walks.shape[1]) / walks.shape[0]
#     for i in range(walks.shape[0]):
#         # taking some random number to be the variate to iterate over
#         benchmark = random.randint(0, walks.shape[0])
#         # iterate over the walks and compute the correlation of every walk with the benchmark walk chosen
#         correlations = np.array([])
#         # add the current mean correlations to the total mean correlation
#         mean_correlations += np.mean(correlations, axis=0)
#     return mean_correlations


"""
calculates the mean correlation in the walks dataset, will be used to benchmark my correlation vs numpy
"""
def mean_auto_correlation(walks):
    # iterate over the walks and compute the correlation of every walk with the benchmark walk chosen
    correlations = np.array([conv_correlation(walks[i], walks[i]) for i in range(walks.shape[0])])
    # return the mean of each column of the correlation matrix you have
    return np.mean(correlations, axis=0)


"""
calculates the mean correlation in the walks dataset, will be used to benchmark my correlation vs numpy
"""
def mean_pearson_auto_correlation(walks):
    # iterate over the walks and compute the correlation of every walk with the benchmark walk chosen
    correlations = np.array([auto_corr_function(walks[i]) for i in range(walks.shape[0])])
    # return the mean of each column of the correlation matrix you have
    return np.mean(correlations, axis=0)


# def mean_auto_correlation(walks):
#     auto_corr = np.zeros(walks.shape[1])
#     for i in range(walks.shape[0]):
#         auto_corr += []


"""
calculates the mean correlation in the walks dataset, will be used to benchmark my correlation vs numpy
"""
def numpy_mean_auto_correlation(walks):
    # iterate over the walks and compute the correlation of every walk with the benchmark walk chosen
    correlations = np.array([np.correlate(walks[i], walks[i], mode='full')[len(walks[i])-1:]
                             for i in range(walks.shape[0])])
    # return the mean of each column of the correlation matrix you have
    return np.mean(correlations, axis=0) / walks.shape[1]
        
    

#BROWNIAN MOTION test for normal generator
mygen = BasicRandomWalk(diffusion, delta_t, NormalGenerator(), steps, const_diffusion)
walks = mygen.generate_walks(amount)
walks, diff = walks[:, :-1], walks[:, -1]
for walk in walks:
    plt.plot([i for i in range(len(walk))], walk)
plt.figure()
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='Brownian Motion')
# tamsd = TA_MSD(walks[1, :])
# # print('ta-msd: {}'.format(tamsd))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))
# # print('msd: {} , ta-msd: {}'.format(msd, tamsd))

# corr = np.correlate(walks[0, :], walks[1, :], mode='full')
# plt.figure()
# plt.plot([i for i in range(walks.shape[1])], walks[0,:], label='walk1')
# plt.plot([i for i in range(walks.shape[1])], walks[1,:], label='walk2')
# plt.legend()
# plt.figure()
# # plt.plot([i for i in range(-walks.shape[1]+1, walks.shape[1])], corr)
# real_corr = corr[int(corr.shape[0]/2):]
# plt.plot([i for i in range(real_corr.shape[0])], real_corr)

# plt.figure()
# auto_corr = np.correlate(walks[0, :], walks[0, :], mode='full')[walks.shape[1]:]
# plt.plot([i for i in range(auto_corr.shape[0])], auto_corr)

# plt.figure()
# walk = walks[0,:]
# size = walk.shape[0]
# auto_corr = [1] + [np.corrcoef(np.array([walk[:-t], walk[t:]]))[0,1] for t in range(1, int(0.9*size))]
# plt.plot([i for i in range(len(auto_corr))], auto_corr)

# plt.figure()
# t = 90
# plt.plot([i for i in range(len(walk[:-t]))], walk[:-t])
# plt.plot([i for i in range(len(walk[t:]))], walk[t:])

# # FBM test positive correlation
# hurst = 0.2
# mygen = FBM(diffusion, delta_t, steps=steps, hurst=hurst)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# # msd = MSD(walks)
# # # plt.plot([i for i in range(len(msd))], msd, label='FBM hurst={}'.format(hurst))
# # # plt.title('msd')
# # print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))


#-------------------------------CORRELATION AND AUTO_CORRELATION TESTS----------------
# # creating the FBM auto correlation function
# alpha = hurst * 2
# anomalous_diff = diffusion / gamma(1+alpha)
# real_auto = np.ones(steps)
# T = steps * delta_t
# times = np.linspace(0, steps*delta_t, steps)
# # the starting time of the correlation is t and the final time is T=steps*delta_t
# for t in np.linspace(steps-1, 0, steps):
#     t = int(t)
#     real_auto[t] = anomalous_diff * (times[t]**alpha + T**alpha - np.abs(T - times[t])**alpha)    
# real_auto = np.flip(real_auto)

# creating the random BM auto correlation function
times = np.linspace(steps*delta_t, 0, steps)
real_auto = diffusion * np.multiply(np.ones(steps), times)

walk1 = walks[0,:]
walk2 = walks[1,:]
# plt.figure()
# plt.plot([i for i in range(len(walk1))], walk1, label='walk1')
# plt.plot([i for i in range(len(walk2))], walk2, label='walk2')

# # my correlation vs real correlation vs numpy correlation
# plt.figure()
# plt.plot
# my_correlation = conv_correlation(walk1, walk2)
# # np_correlation = np.correlate(walk1, walk2, mode='full')[len(walk1)-1:] / steps
# plt.plot([i for i in range(my_correlation.shape[0])], my_correlation, label='my correlation')
# # plt.plot([i for i in range(np_correlation.shape[0])], np_correlation, label='numpy correlation')
# plt.plot([i for i in range(real_auto.shape[0])], real_auto, label='real autocorrelation')
# plt.legend()

# # my mean correlation vs real mean correlation vs numpy mean correlation
# plt.figure()
# plt.plot
# my_correlation = mean_correlation(walks)
# np_correlation = numpy_mean_correlation(walks)
# plt.plot([i for i in range(my_correlation.shape[0])], my_correlation, label='my correlation')
# plt.plot([i for i in range(np_correlation.shape[0])], np_correlation, label='numpy correlation')
# plt.legend()

# print(mean_correlation(walks))

# my mean auto correlation vs real mean auto correlation vs numpy mean auto correlation
plt.figure()
plt.plot
my_correlation = mean_auto_correlation(walks)
# np_correlation = numpy_mean_auto_correlation(walks)
# pearson_correlation = mean_pearson_auto_correlation(walks)
plt.plot(times, my_correlation, label='my autocorrelation')
# plt.plot([i for i in range(pearson_correlation.shape[0])], pearson_correlation, label='pearson autocorrelation')
# plt.plot([i for i in range(np_correlation.shape[0])], np_correlation, label='numpy autocorrelation')
plt.plot(times, real_auto, label='real autocorrelation')
plt.legend()


# numpy_covariance = np.cov(walk1, walk2)
# my_covariance = covariance(walk1, walk2)
# print('numpy covariance diff: {}'.format(round(numpy_covariance[0,1] - my_covariance, 6)))

# numpy_corrcoef = np.corrcoef(walk1, walk2)
# my_correlation = correlation(walk1, walk2)
# print('numpy corrcoef diff: {}'.format(round(numpy_corrcoef[0,1] - my_correlation, 6)))

# series1, series2 = pd.Series(walk1), pd.Series(walk2)
# pandas_correlation = series1.corr(series2)
# print('pandas correlation diff: {}'.format(round(pandas_correlation - my_correlation, 6)))

# lag = 1
# pandas_autocorr = series1.autocorr(lag=lag)
# my_autocorr = autocorrelation(walk1, lag=lag)
# print('pandas auto correlation diff: {}'.format(round(pandas_autocorr - my_autocorr, 6)))

# np_autocorr = numpy_auto_corr_func(walk1)
# my_autocorr = auto_corr_function(walk1)
# pd_autocorr = pandas_auto_corr_func(series1).to_numpy()
# plt.figure()
# # plt.figure()
# pd.plotting.autocorrelation_plot(series1, label='pandas autocorrelation')
# plt.plot([i for i in range(np_autocorr.shape[0])], np_autocorr, label='numpy autocorrelation')
# plt.plot([i for i in range(my_autocorr.shape[0])], my_autocorr, 'r+', label='my autocorrelation')
# # plt.plot([i for i in range(pd_autocorr.shape[0])], pd_autocorr, label='pandas autocorrelation manual')
# plt.plot([i for i in range(real_auto.shape[0])], real_auto, label='real autocorrelation')
# plt.legend()

# #------------------------------- TIME TESTS -----------------------------------
# # numpy vs pandas vs my correlation, not taking into account the time it takes to convert between numpy and pandas
# # since the dataset is given in pandas anyway
# np_start = time.time()
# for i in range(1000):
#     numpy_corrcoef = np.corrcoef(walk1, walk2)
# np_end = -(np_start - time.time())

# my_start = time.time()
# for i in range(1000):
#     my_correlation = correlation(walk1, walk2)
# my_end = -(my_start - time.time())

# series1, series2 = pd.Series(walk1), pd.Series(walk2)
# pd_start = time.time()
# for i in range(1000):
#     pandas_correlation = series1.corr(series2)
# pd_end = -(pd_start - time.time())
# print('CORRELATION: numpy time: {}    my time: {}    pandas time: {}'.format(np_end, my_end, pd_end))


# # pandas vs me vs numpy autocorrelation
# temp_walk = pd.DataFrame(walk1)
# my_start = time.time()
# for i in range(100):
#     # since the preprocess will constantly convert rows to numpy, I will add it to the calculation
#     temp_walk.to_numpy()
#     my_autocorr = auto_corr_function(walk1)
# my_end = -(my_start - time.time())

# np_start = time.time()
# for i in range(100):
#     # since the preprocess will constantly convert rows to numpy, I will add it to the calculation
#     temp_walk.to_numpy()
#     np_autocorr = numpy_auto_corr_func(walk1)
# np_end = -(np_start - time.time())

# series1 = pd.Series(walk1)
# pd_start = time.time()
# for i in range(100):
#     pandas_autocorr = pandas_auto_corr_func(series1)
# pd_end = -(pd_start - time.time())
# print('AUTO-CORRELATION: my time: {}    pandas time: {}    numpy time: {}'.format(my_end, pd_end, np_end))



# # corr = np.correlate(walks[0, :], walks[1, :], mode='full')
# # plt.figure()
# # plt.plot([i for i in range(walks.shape[1])], walks[0,:], label='walk1')
# # plt.plot([i for i in range(walks.shape[1])], walks[1,:], label='walk2')
# # plt.legend()
# # plt.figure()
# # # plt.plot([i for i in range(-walks.shape[1]+1, walks.shape[1])], corr)
# # real_corr = corr[int(corr.shape[0]/2):]
# # plt.plot([i for i in range(real_corr.shape[0])], real_corr)

# # plt.figure()
# # auto_corr = np.correlate(walks[0, :], walks[0, :], mode='full')[walks.shape[1]:]
# # plt.plot([i for i in range(auto_corr.shape[0])], auto_corr)

# # plt.figure()
# # size = walk.shape[0]
# # auto_cov = [1] + [np.cov(np.array([walk[:-t], walk[t:]]))[0,1] for t in range(1, size-1)]
# # plt.plot([i for i in range(len(auto_cov))], auto_cov)
# # plt.title('covariance')

# plt.figure()
# size = walk.shape[0]
# auto_corr = [1] + [np.corrcoef(np.array([walk[:-t], walk[t:]]))[0,1] for t in range(10, size-1)]
# plt.plot([i for i in range(len(auto_corr))], auto_corr)
# plt.title('pearsons auto correlation')

# plt.figure()
# pd.plotting.autocorrelation_plot(pd.DataFrame(walk))

# plt.figure()
# series = pd.Series(walk)
# auto_corr = [series.autocorr(t) for t in range(len(walk) - 1)]
# plt.plot([i for i in range(len(auto_corr))], auto_corr)
# plt.title('pearsons autocorrelation')

# plt.figure()
# t = 3
# plt.plot([i for i in range(len(walk[:-t]))], walk[:-t])
# plt.plot([i for i in range(len(walk[t:]))], walk[t:])


# plt.figure()
# x = pd.Series(np.linspace(0, 2 * np.pi, 100))
# y = pd.DataFrame(np.sin(x))
# plt.plot(x,y)
# print(y.corr(x))








