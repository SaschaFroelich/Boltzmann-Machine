#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:46:15 2020

Implement a Hebbian learning algorithm (created by Sascha) for a Boltzmann
machine

change in connection strength w_ij = a*exp(-b*w_ij)s_i*s_j-a/3*exp(b*w_ij)(1-s_i*s_j)

In a purely random set of states, the probability of si*sj = 1 is 1/3 the prob of si*sj=0.
Therefore, we have a/3 in the second part of the term after the minus sign.

@author: sascha
"""

import numpy as np
import itertools
from tqdm import *
import matplotlib.pyplot as plt
import scipy as sci
import seaborn as sns
import sys
import BM

#%%
np.random.seed(123)

N = 5
omega = 0.5

# which omega do you want to observe?
iobs = 1
jobs = N-1
omega_hist = []

# Crate all 1024 random state vectors for N neurons
lst = [list(i) for i in itertools.product([0, 1], repeat=N)]

# Assign them all a probability to occur in the real world.
# Sample from fat-tailed Poisson distro
probs = np.random.poisson(lam=2, size=len(lst))
probs = probs/sum(probs)
#probs = np.sort(probs)
plt.scatter(range(len(probs)),probs)
plt.show()

#%%
a = 1/100
b = 0.1
# initialise connectivities randomly > 0
W = np.random.uniform(low=0.0, high=1.0, size=(N,N))


# Iterate often enough for the least likely state to occur 10 times (on average)
iterations = 40_000

W_hist = np.zeros((N,N,iterations))


for iterstep in tqdm(range(iterations)):
    
    W_hist[:,:,iterstep] = W.copy()
    
    # Choose a state randomly
    state_idx = np.random.choice(np.arange(0, len(lst)), p=probs)
    state = lst[state_idx]
    
    #print(state_idx)
    
    for i in range(N):
        W[i,i] = 0
        for j in range(i+1, N):
            delta_omega = a*np.exp(-b*W[i,j])*state[i]*state[j] - \
            a/3*np.exp(b*W[i,j])*(1-state[i]*state[j])
            W[i,j] += delta_omega
            W[j,i] = W[i,j]
            
means = W_hist[:,:,2000:].mean(axis=2)
standdevs = W_hist[:,:,2000:].std(axis=2)
#%%
mns=[]
stds=[]
for i in range(N):
    for j in range(N):
        if i is not j:
            mns.append(means[i,j])
            stds.append(standdevs[i,j])
#%%

# The means are the means along W[i,j,:]
sns.kdeplot(mns)
plt.title('means of w_ij')
plt.show()
sns.kdeplot(stds)
plt.title('stds of w_ij')
plt.show()


print("Ratio between std of means and mean odf std (std of means should be larger than the averge std) -> > 1")
print(np.array(mns).std()/np.array(stds).mean())

#res = sci.stats.shapiro(W_hist[1,4][2000:])
#print(res)

#%%

for i in range(N):
    for j in range(i+1, N):
        plt.figure()
        plt.plot(W_hist[i,j,:])
        plt.ylim([-6,2])
        plt.show()
        
#%% Boltzmann without biases
    
# List all possible states for the self.N visible neurons
all_possible_states = [list(i) for i in itertools.product([0, 1], repeat=N)]

# Compute <s_i s_j> in the data
expect_sisj = np.zeros((N,N))

for idx, state in enumerate(data):
    for si, value in enumerate(state):
        if value == 1:
            expect_sisj[si,:] += state
            
np.fill_diagonal(expect_sisj, 0)

expect_sisj = expect_sisj/len(data)

#%%
all_possible_states = [list(i) for i in itertools.product([0, 1], repeat=N)]
