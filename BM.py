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
        # state has to be a 1D or 2D list
        # returns 1D or 2D list
#        breakpoint()
        if np.array(state).ndim == 2:
            E = []
            for s in state:
                E.append(self.energy(s))
        
            return E
            
        else:
            if type(state) == list:
                state = np.array(state)
            
            E = -state@self.bias
            for jdx in range(len(state)):
                for idx in range(jdx):
                     E -= state[idx]*state[jdx]*self.W[idx,jdx]
                     
            return E
    
    def ncr(self, n, r):
        # nchooser(n,r)
        from functools import reduce
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

    def logistic(self, z):
        return 1/(1+np.exp(-z/self.T))

    def inpt(self,state):
        # COmputes inputs to the current states
        return self.bias + self.W@state
    
    def iterate(self, no_iters, savehist = False, suppress_output = False):
        # Runs the generative model to create states
        # If BM did not run before, it will need some time to settle into a
        # stable state distribution
        
        
        from tqdm import tqdm
        
        state = self.state
        
        if not suppress_output:
            print("============================================")
            print("Initial state:")
            print(state)

        if savehist:
            state_hist = [0]*no_iters

        for iterdx in tqdm(range(no_iters)):
            for i in range(self.N):
                z = self.bias + self.W@state
                prob_active = self.logistic(z)

                if np.random.rand(1) > prob_active[i]:
                    state[i] = 0
                else:
                    state[i] = 1

            if savehist:
                state_hist[iterdx] = state.copy().tolist()

            # Compute energy of state vector
            E = self.energy(state)
        
        
        if not suppress_output:
            print("End of iteration")
        self.state = state

        if savehist:
            return state, state_hist
        else:
            return state
        
    def learn(self, data, it = 1000, alpha = 0.1):
        # it = no. of iterations (iterates it times over all data)
        # Data have to be a 2D list, where a (first entry of list) row is one state vector of the visible units
        # alpha: learning rate
        
        """
        # Generate all possible states for later estimation of <sisj>_model
        # List all possible states for the self.N visible neurons
        import itertools
        all_possible_states = [list(i) for i in itertools.product([0, 1], \
                               repeat=self.N)]
        """
    
        W_hist = np.zeros((self.N,self.N,it))
        expect_sisj_data, expect_si_data = self.expect_sisj_data(data)
        
        from tqdm import tqdm
        for i in tqdm(range(it)):
            expect_sisj_model, expect_si_model = self.expect_sisj_model()
            self.W = self.W + alpha*(expect_sisj_data - expect_sisj_model)
            self.bias = self.bias + alpha*(expect_si_data - expect_si_model)
            
            # W_hist[:,:,i] = self.W
                
        return self.W, self.bias
        
    def expect_sisj_data(self, data):
        # returns expect_sisj_data
        # data has to be a list of dimensions no_vectors x no_units (visible units)

        ## Compute <s_i s_j> and <s_i> in the data
        expect_sisj_data = np.zeros((self.N,self.N))
        expect_si_data = np.zeros(self.N)
        
        
        for idx, state in enumerate(data):
            expect_si_data += state
            for si, value in enumerate(state):
                if value == 1:
                    expect_sisj_data[si,:] += state
                    
        np.fill_diagonal(expect_sisj_data, 0)
        expect_sisj_data = expect_sisj_data/len(data)
        expect_si_data = expect_si_data/len(data)
                
        return expect_sisj_data, expect_si_data
        
    def expect_sisj_model(self):
        ## Compute <s_i s_j> and <si> in the model
        expect_sisj_model = np.zeros((self.N,self.N))
        expect_si_model = np.zeros(self.N)
        
        # Z is denominator in Boltzmann distro (i.e. partition function)
        Z = sum(self.energy(all_possible_states))
        
        for i in range(self.N):
            matches_si = [state for state in all_possible_states if state[i] == 1]
            energies_si = np.array(self.energy(matches_si))
            expect_si_model[i] = np.exp(-energies_si).sum() / Z
            
            for j in range(i+1, self.N):
                matches = [state for state in all_possible_states if \
                           state[i] == 1 and state[j]==1]
                
                energies = np.array(self.energy(matches))
                expect_sisj_model[i,j] = np.exp(-energies).sum() / Z
                expect_sisj_model[j,i] = expect_sisj_model[i,j].copy()
                
        return expect_sisj_model, expect_si_model