# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:09:50 2020

@author: gedadav
"""

import Generator
from RandomWalks import BasicRandomWalk, CTRW, FBM
from RandomWalks import TA_MSD, MSD, ensemble_TA_MSD
import numpy as np
import matplotlib.pyplot as plt
from Generator import NormalGenerator, UniformGenerator, ExponGenerator, LevyGenerator, PowerLawGenerator, SymmetricExponGenerator
from stochastic.processes.continuous import FractionalBrownianMotion, BrownianMotion
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style defaults and set the default figure size
sns.set()
sns.set_context('paper', font_scale=2.5)


# creating a constant diffusion Brownian motion to test
diffusion = 0.3
const_diffusion = True
delta_t = 0.05
steps = int(1e2)
amount = int(1e4)
plt.figure()
plt.xlabel('t')
plt.ylabel('x')

# #BROWNIAN MOTION test for normal generator
# mygen = BasicRandomWalk(diffusion, delta_t, NormalGenerator(), steps, const_diffusion)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='msd')
# plt.xlabel('time')
# plt.ylabel('MSD values')
# # getting a ta-msd ensemble with 0.1 of the total amount of walks
# # ensemble_tamsd = ensemble_TA_MSD(walks, ensemble=100)
# # plt.plot([i for i in range(len(ensemble_tamsd))], ensemble_tamsd, label='ensemble ta-msd')
# # plt.legend()
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(delta_t*steps)))
# # print('ta-msd: {}, diffusion: {}'.format(ensemble_tamsd[-1], ensemble_tamsd[-1]/(delta_t*steps)))

 
# #BROWNIAN MOTION test for symmetric exponential
# mygen = BasicRandomWalk(diffusion, delta_t, SymmetricExponGenerator(), steps, const_diffusion)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='Brownian Motion')
# # tamsd = TA_MSD(walks[1, :])
# # print('ta-msd: {}'.format(tamsd))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))
# # print('msd: {} , ta-msd: {}'.format(msd, tamsd))
 

# #CTRW test exponential waiting times
# # time generator mean = 1/(100*delta_t)
# # TODO add the calculation here to depend on delta
# mygen = CTRW(diffusion, delta_t, steps=steps, time_generator=ExponGenerator(100*delta_t))
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='CTRW exponential waiting times')
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))

# #CTRW test uniform waiting times
# mygen = CTRW(diffusion, delta_t, NormalGenerator(), steps=steps, time_generator=UniformGenerator(2*delta_t, 10*delta_t))
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='CTRW uniform waiting times')
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))

# #CTRW test powerlaw waiting times subdiffusion
# alpha = 2.15
# mygen = CTRW(diffusion, delta_t,  NormalGenerator(), PowerLawGenerator(alpha, delta_t), steps,
#                                                 const_diffusion, None)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='CTRW alpha={}'.format(alpha))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))

# #CTRW test powerlaw waiting times
# alpha = 3.2
# mygen = CTRW(diffusion, delta_t,  NormalGenerator(), PowerLawGenerator(alpha, delta_t), steps,
#                                                 const_diffusion, None)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='CTRW alpha={}'.format(alpha))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))


# # FBM test negative correlation
# exp = 0.1
# mygen = FBM(diffusion, delta_t, steps=steps, hurst=exp)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='FBM hurst={}'.format(exp))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))

# # FBM test positive correlation
# exp = 0.9
# mygen = FBM(diffusion, delta_t, steps=steps, hurst=exp)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd, label='FBM hurst={}'.format(exp))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))

# # FBM test Brownian motion
# mygen = FBM(diffusion, delta_t, steps=steps, hurst=0.5)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# plt.figure()
# plt.xlabel('time')
# plt.ylabel('MSD values')
# msd = MSD(walks)
# plt.plot([i for i in range(len(msd))], msd)
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))

# #Levy test
# mygen = BasicRandomWalk(diffusion, delta_t, LevyGenerator(1e-7), steps, const_diffusion)
# walks = mygen.generate_walks(amount)
# walks, diff = walks[:, :-1], walks[:, -1]
# for walk in walks:
#     plt.plot([i for i in range(len(walk))], walk)
# msd = MSD(walks)
# plt.figure()
# plt.plot([i for i in range(len(msd))], msd)
# tamsd = TA_MSD(walks[1, :])
# print('ta-msd: {}'.format(tamsd))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))
# # print('msd: {} , ta-msd: {}'.format(msd, tamsd))

#Levy powerlaw test
# alpha > 2 : subdiffusion
alpha = [2,3]
mygen = BasicRandomWalk(diffusion, delta_t, PowerLawGenerator(alpha, delta_t), steps, const_diffusion)
walks = mygen.generate_walks(amount)
walks, diff = walks[:, :-1], walks[:, -1]
for walk in walks:
    plt.plot([i for i in range(len(walk))], walk)
plt.xlabel('t')
plt.ylabel('x')
msd = MSD(walks)
plt.figure()
plt.plot([i for i in range(len(msd))], msd)

# # tamsd = TA_MSD(walks[1, :])
# # print('ta-msd: {}'.format(tamsd))
# print('msd: {}, diffusion: {}'.format(msd[-1], msd[-1]/(2*delta_t*steps)))
# # print('msd: {} , ta-msd: {}'.format(msd, tamsd))

#plt.title('MSD')
plt.xlabel('time')
plt.ylabel('MSD values')
#plt.legend()
    
"""
# tests using the stochastic processes library
from stochastic.processes.continuous import BrownianMotion

steps = 20
walker_amount = 1000
delta_t = 0.05
bm = BrownianMotion(t=delta_t*steps)
samples = bm.sample(100)
plt.plot(bm.times(steps), bm.sample(steps))
plt.show()
from stochastic.processes.continuous import FractionalBrownianMotion


fbm = FractionalBrownianMotion(hurst=0.3, t=delta_t*steps)
s = fbm.sample(steps)
times = fbm.times(steps)

plt.plot(times, s)
plt.show()
"""