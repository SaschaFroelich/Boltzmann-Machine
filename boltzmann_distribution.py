#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:09:17 2020

@author: sascha
"""

# Distance from actual Boltzmann Distro

import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

import operator as op
from functools import reduce

import BM

#%%
no_iterations = np.linspace(10,10e2,100)

average_distance = []

for iters in no_iterations:
    print("The no of iters is %d"%int(iters))
    bm = BM.BoltzmannM(10, init_state=np.ones(10))
    boltzmann_occ, rel_occ_in_uniques, E = bm.iterate(int(iters))
    dist = abs(boltzmann_occ-rel_occ_in_uniques).sum()/len(rel_occ_in_uniques)
    average_distance.append(dist)
    
    
#%%
bm = BM.BoltzmannM(10, init_state=np.ones(10))
boltzmann_occ, rel_occ_in_uniques, E = bm.iterate(100)

#%%
plt.figure()
plt.plot(no_iterations,average_distance)
plt.ylabel('Distance [a.u.]')
plt.title('Average distance between Boltzmann distribution and actual state vector distribution at the end of the iterations')
plt.xlabel('No of iterations')
plt.savefig('/home/sascha/Desktop/PrivateProjects/Boltzmann Machine/average_distance.svg')
plt.show()