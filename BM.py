#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:41:56 2020

@author: sascha
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import operator as op
from functools import reduce

class BoltzmannM:
    # **kwargs: "W" (connectivity matrix), "bias" (bias-vector), "init_ state" (initial 
    # state vector), T (system temp)
    def __init__(self, no_neurons, **kwargs):
        # Initialise values N, W, bias, init_state
        
        self.N = no_neurons
        
        if 'W' in kwargs:
            self.W = kwargs['W']
        else:
            # Generate random connectivity matrix W
            max_strength = 1/self.N

            W = np.zeros([self.N,self.N])
            for n1 in range(self.N):
                for n2 in range(n1,self.N):
                    if (n1 != n2):
                         W[n1,n2] = np.random.rand(1)*max_strength
                         W[n2,n1] = W[n1,n2]
            self.W = W
            
        if 'bias' in kwargs:
            self.bias = kwargs['bias']
        else: 
            # Create random bias vector from uniform distro on [shift, scale + shift]
            scale = 2
            shift = -2
            self.bias = np.random.rand(self.N)*scale + shift
        
        if 'init_state' in kwargs:
            self.state = kwargs['init_state']
        else: 
            # Create a random initial state:
            self.state = np.random.randint(0,high=2,size=self.N)
            
        if 'T' in kwargs:
            self.T = kwargs['T']
        else: 
            self.T = 1
        
    def energy(self, state):
        E = -state@self.bias
        for jdx in range(len(state)):
            for idx in range(jdx):
                 E -= state[idx]*state[jdx]*self.W[idx,jdx]

        return E
    
    def ncr(self, n, r):
        # nchooser(n,r)
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

    def logistic(self, z):
        return 1/(1+np.exp(-z/self.T))

    def inpt(self,state):
        # COmputes inputs to the current states
        return self.bias + self.W@state
    
    def iterate(self, no_iters, save_history = False):
        from tqdm import tqdm
        
        print("============================================")
        print("Doing %d iterations"%no_iters)

        total_no_states = 0
        for n in range(self.N+1):
            #n is the no. of "on" states
            total_no_states += self.ncr(self.N,n)

        print("There is a total of %d different possible states for %d neurons"\
              %(total_no_states,self.N))

        print("Initial state:")
        state = self.state
        print(state)

        if save_history:
            state_hist = []

        for iter in tqdm(range(no_iters)):
            for i in range(self.N):
                z = self.bias + self.W@state
                prob_active = self.logistic(z)

                if np.random.rand(1) > prob_active[i]:
                    state[i] = 0
                else:
                    state[i] = 1


            # Compute energy of state vector
            E = self.energy(state)

        print("\nEnd state:")
        print(state)
        
        print("End of iteration.")
        print("============================================")
        self.state = state

        if save_history:
            state_hist.append(state.copy())
            
            return state, state_hist
        else:
            return state