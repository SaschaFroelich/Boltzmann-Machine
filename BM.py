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
    def __init__(self, no_vis, no_hidden = 0, **kwargs):
        # Initialise values N_vis,, N_hidden W, bias, init_state
        
        self.N_vis = no_vis
        self.N_hidden = no_hidden
        
        if 'W' in kwargs:
            self.W = kwargs['W']
        else:
            # Generate random connectivity matrix W
            max_strength = 1/(self.N_vis + self.N_hidden)

            W = np.zeros([self.N_vis+self.N_hidden,self.N_vis+self.N_hidden])
            for n1 in range(self.N_vis+self.N_hidden):
                for n2 in range(n1,self.N_vis+self.N_hidden):
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
            self.bias = np.random.rand(self.N_vis+self.N_hidden)*scale + shift
        
        if 'init_state' in kwargs:
            self.state = kwargs['init_state']
        else: 
            # Create a random initial state:
            self.state = np.random.randint(0,high=2,size=self.N_vis+\
                                           self.N_hidden)
            
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

    def logistic(self, z):
        return 1/(1+np.exp(-z/self.T))

    def inpt(self,state):
        # COmputes inputs to the current states
        return self.bias + self.W@state
    
    def iterate(self, epochs, savehist = False, suppress_output = False):
        # Runs the generative model to create states
        # If BM did not run before, it will need some time to settle into a
        # stable state distribution
        # 1 epoch corresponds to 1 iteration step where all units get updated once
        
        
        from tqdm import tqdm
        
        state = self.state
        
        if not suppress_output:
            print("============================================")
            print("Initial state:")
            print(state)

        if savehist:
            state_hist = [0]*epochs

        for e in tqdm(range(epochs)):
            z = self.bias + self.W@state
            prob_active = self.logistic(z)
            
            for i in range(self.N_vis):
                if np.random.rand(1) > prob_active[i]:
                    state[i] = 0
                else:
                    state[i] = 1

            if savehist:
                state_hist[e] = state.copy().tolist()
        
        if not suppress_output:
            print("End of iteration")
            
        self.state = state

        if savehist:
            return state_hist
        else:
            return state
        
    def learn(self, data, it = 1000, alpha = 0.1):
        # it = no. of iterations (iterates it times over all data)
        # Data have to be a 2D list, where a (first entry of list) row is one state vector of the visible units
        # alpha: learning rate
    
        # W_hist = np.zeros((self.N_vis,self.N_vis,it))
        
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

        """Added self.N_hidden, but still have to implement learning with N_hidden"""

        ## Compute <s_i s_j> and <s_i> in the data
        expect_sisj_data = np.zeros((self.N_vis+self.N_hidden,self.N_vis+\
                                     self.N_hidden))
        
        expect_si_data = np.zeros(self.N_vis+self.N_hidden)
        
        
        for idx, state in enumerate(data):
            expect_si_data += state
            for si, value in enumerate(state):
                if value == 1:
                    expect_sisj_data[si,:] += state
                    
        np.fill_diagonal(expect_sisj_data, 0)
        expect_sisj_data = expect_sisj_data/len(data)
        expect_si_data = expect_si_data/len(data)
                
        return expect_sisj_data, expect_si_data
        
    def expect_sisj_model(self, iterations = 100_000):
        ## Compute <s_i s_j> and <si> in the model
        expect_sisj_model = np.zeros((self.N_vis,self.N_vis))
        expect_si_model = np.zeros(self.N_vis)
        
        _, statehist = self.iterate(iterations, savehist = True)
        
        """Compute <s_i>"""
        for i in range(self.N_vis):
            matches_si = [state for state in statehist if state[i] == 1]
            expect_si_model[i] = len(matches_si)/len(statehist)
            
            """Compute <s_is_j>"""
            for j in range(i+1, self.N_vis):
                matches = [state for state in statehist if \
                           state[i] == 1 and state[j]==1]
                
                expect_sisj_model[i,j] = len(matches)/len(statehist)
                expect_sisj_model[j,i] = expect_sisj_model[i,j].copy()

        return expect_sisj_model, expect_si_model
    
    
"""restricted Boltzmann Machine"""
class rBoltzmannM():
    pass