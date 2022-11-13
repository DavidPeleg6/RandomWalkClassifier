# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pylab
import random
import math
from scipy.stats import levy, randint, powerlaw
from Generator import *
import random
import powerlaw
import seaborn as sns

# Use seaborn style defaults and set the default figure size
sns.set()
sns.set_context('paper', font_scale=2.5)

# increase plot size
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size


# # exponential generator test
# my_gen = ExponGenerator(lamda=(1))
# seq = [my_gen.generate_variate() for i in range(10000)]
# print(my_gen.exp.stats())
# print('mean {}'.format(sum(seq) / 10000))
# plt.hist(seq, 100)
# plt.show()

# # normal generator test
# my_gen = NormalGenerator(mean=0, std=1e-2)
# seq = [my_gen.generate_variate() for i in range(10000)]
# plt.hist(seq, 50)
# plt.show()

# # normal generator test
# seq = [random.random() for i in range(10000)]
# plt.hist(seq, 100)
# plt.show()


# # symmetric exponential generator test
# my_gen = SymmetricExponGenerator()
# seq = [my_gen.generate_variate() for i in range(10000)]
# plt.hist(seq, 100)
# print('mean  = {} variance = {}'.format(np.mean(seq), np.var(seq)))
# plt.show()


# uniform generator tests
# my_gen = UniformGenerator(0.49, 0.51)
# seq = [my_gen.generate_variate() for i in range(10000)]
# plt.hist(seq, 100)
# plt.show()


# power law generator tests
alpha = 3
delta = 1e-5
my_gen = PowerLawGenerator(alpha, delta)
seq = [my_gen.generate_variate() for i in range(1000)]
results = powerlaw.Fit(seq)
print(results.power_law.alpha)
print(results.power_law.xmin)
R, p = results.distribution_compare('power_law', 'lognormal')
plt.hist(seq, 100)
plt.title('power law test')
plt.show()

# # power law generator2 tests
# alpha = 3
# delta = 0.05
# xmin = ((1/delta) * (alpha-1)) ** (1/(1-alpha))
# my_gen = PowerLawGenerator(alpha, delta)
# seq = [my_gen.generate_variate() for i in range(1000)]
# results = powerlaw.Fit(seq)
# print('real alpha: {} predicted alpha: {}'.format(alpha, results.power_law.alpha))
# print('real xmin: {} predicted xmin: {}'.format(xmin, results.power_law.xmin))
# R, p = results.distribution_compare('power_law', 'lognormal')
# plt.hist(seq, 100)
# plt.title('power law test')
# plt.show()

# # power law generator tests2
# alpha = 8
# xmin = (alpha-1) ** (1/(1-alpha))
# delta = 0.05
# shape = (5, 10000)
# my_gen = PowerLawGenerator(alpha)
# tens = my_gen.generate_tensor(shape, False)
# seq1 = tens[2, :]
# results = powerlaw.Fit(seq1)
# print('real alpha: {} predicted alpha: {}'.format(alpha, results.power_law.alpha))
# print('real xmin: {} predicted xmin: {}'.format(xmin, results.power_law.xmin))
# R, p = results.distribution_compare('power_law', 'lognormal')
# plt.hist(seq1, 100)
# plt.title('power law sequence1')
# plt.show()

# seq1 = tens[0, :]
# results = powerlaw.Fit(seq1)
# print('real alpha: {} predicted alpha: {}'.format(alpha, results.power_law.alpha))
# print('real xmin: {} predicted xmin: {}'.format(xmin, results.power_law.xmin))
# R, p = results.distribution_compare('power_law', 'lognormal')
# plt.hist(seq1, 100)
# plt.title('power law sequence2')
# plt.show()



"""
# exponential generator test
my_gen = SymmetricExponGenerator()
seq = []
for i in range(100):
    seq.extend(list(my_gen.generate_tensor([100,1],torch_tensor=False)))
plt.hist(seq, 100)
plt.show()
"""

"""
# levy generator test
my_gen = LevyGenerator()
seq = [None] * 10000
for i in range(len(seq)):
    temp = my_gen.generate_variate()
    seq[i] = temp if temp < 10 else 0
# seq = [my_gen.generate_variate() for i in range(10000)]
plt.hist(seq, 100)
plt.show()
"""
"""
# Define parameters for the walk
dims = 1
step_n = 10000
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))
# Simulate steps in 1D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]
# Plot the path
fig = plt.figure(figsize=(8,4),dpi=200)
ax = fig.add_subplot(111)
ax.scatter(np.arange(step_n+1), path, c='blue',alpha=0.25,s=0.05)
ax.plot(path,c='blue',alpha=0.5,lw=0.5,ls='-',)
ax.plot(0, start, c='red', marker='+')
ax.plot(step_n, stop, c='black', marker='o')
plt.title('1D Random Walk')
plt.tight_layout(pad=0)
plt.show()
"""
"""
# Define parameters for the walk
dims = 2
step_n = 10000
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))
# Simulate steps in 2D
step_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]
# Plot the path
fig = plt.figure(figsize=(8,8),dpi=200)
ax = fig.add_subplot(111)
ax.scatter(path[:,0], path[:,1], c='blue', alpha=0.25, s=0.05);
ax.plot(path[:,0], path[:,1], c='blue', alpha=0.5, lw=0.25, ls='-');
ax.plot(start[:,0], start[:,1],c='red', marker='+')
ax.plot(stop[:,0], stop[:,1],c='black', marker='o')
plt.title('2D Random Walk')
plt.tight_layout(pad=0)
plt.show()
"""
"""
# Python code for 2D random walk.


# defining the number of steps
n = 1000

# creating two array for containing x and y coordinate
# of size equals to the number of size and filled up with 0's
x = np.zeros(n)
y = np.zeros(n)
# x = np.linspace(0, 200, n)
# y = np.linspace(0, 200, n)
frozen_levy = levy()

# filling the coordinates with random variables
for i in range(1, n):
    angle = random.random() * 2 * math.pi
    size = frozen_levy.rvs()
    if size < 600:
        x[i] = size * math.cos(angle)
        y[i] = size * math.sin(angle)


# plotting stuff:
pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.plot(np.cumsum(x), np.cumsum(y))
# pylab.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
pylab.show()
"""
