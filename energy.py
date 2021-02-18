#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:15:20 2020

@author: sascha
"""

# Development of energy when running BM $Nall times for $bmiters iterations
Nall = 20
bmiters = 1000

# Initialise BM with specific parameters
Es = np.array(np.zeros([Nall,bmiters]))

for i in range(Nall):
    bm = BM.BoltzmannM(10, init_state=np.ones(10), bias=np.random.rand(10), T = 0.01)
    boltzmann_occ, rel_occ_in_uniques, E = bm.iterate(bmiters)
    Es[i,:] = E

plt.figure()
plt.plot(Es.mean(axis=0))
plt.show()

#%%

bm = BM.BoltzmannM(10, init_state=np.ones(10), bias=np.random.rand(10), T = 0.01)
boltzmann_occ, rel_occ_in_uniques, E = bm.iterate(1000)