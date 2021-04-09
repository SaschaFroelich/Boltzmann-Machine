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
    def __init__(self, no_vis, no_hid = 0, restr = 0, **kwargs):
        """" 
        INPUT
            no_vis  :   no of visible neurons
            
            OPTIONAL
            no_hid   : no. of hidden neurons (default 0)
            restr       : BOOL value (1: restricted BM, 0: non-restricted BM (Default))
            W           : (symmetric matrix with diagonal = 0) matrix connection weights. 
                          Size (no_vis+no_hid)*(no_vis+no_hid)
                          (Default: 
                          [non-restricted BM]
                          each weight initialised uniformly randomly between 0 and 1/(no_vis+no_hid). 
                          Diagonal elements are 0.
                            
                          [restricted BM, see above]
                          connections *only* between visible and non-visible units:
                          only lower-left rectangle and upper-right rectangle are non-zero)
            bias        : bias vector (size (no_vis+no_hid)x1) (Default:
                          each bias initialised uniformly randomly between -1 and 1.)
            init_state  : (vector of size (no_vis + no_hid)x1) initial state of the BM. 
        
        """
        # Initialise values N_vis,, N_hid W, bias, init_state
        
        self.N_vis = no_vis
        self.N_hid = no_hid
        self.restricted = restr
        
        if 'W' in kwargs:
            self.W = kwargs['W']
        else:
            # Generate random connectivity matrix W
            max_strength = 1/(self.N_vis + self.N_hid)
            W = np.zeros([self.N_vis+self.N_hid,self.N_vis+self.N_hid])
            
            if restr:
                for n1 in range(self.N_vis+self.N_hid):
                    for n2 in range(n1,self.N_vis+self.N_hid):
                        if (n1 <= self.N_vis and n2 > self.N_vis):
                             W[n1,n2] = np.random.rand(1)*max_strength
                             W[n2,n1] = W[n1,n2]
                
            else:
                
                for n1 in range(self.N_vis+self.N_hid):
                    for n2 in range(n1,self.N_vis+self.N_hid):
                        if (n1 != n2):
                             W[n1,n2] = np.random.rand(1)*max_strength
                             W[n2,n1] = W[n1,n2]
            self.W = W
            
        if 'bias' in kwargs:
            self.bias = kwargs['bias']
        else:
            # Create random bias vector from uniform distro on [shift, scale + shift]
            scale = 2
            shift = -1
            self.bias = np.random.rand(self.N_vis+self.N_hid)*scale + shift
        
        if 'init_state' in kwargs:
            self.state = kwargs['init_state']
        else: 
            # Create a random initial state:
            self.state = np.random.randint(0,high=2,size=self.N_vis+\
                                           self.N_hid)
        
    def energy(self, state):
        # state has to be a 1D or 2D list
        # returns 1D or 2D list
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

    def logistic(self, z, T=1):
        #return 1/(1+np.exp(-z/T - np.max(-z/T))/np.exp(-np.max(-z/T)))
        return 1/(1+np.exp(-z/T))

    def inpt(self,state):
        # COmputes inputs to the current states
        return self.bias + self.W@state
    
    def iterate(self, epochs = 1, T = 1):
        """output
        states: last network state after iteration (in case of 1 epoch).
                history of states during iteration (in case of epochs > 1)
        """
        #from tqdm import tqdm
        # Runs the generative model to create states
        # If BM did not run before, it will need some time to settle into a
        # stable state distribution
        # 1 epoch corresponds to 1 iteration step where all units get updated once        
        state = self.state
        #print(state)

        states = []
        for e in range(epochs):
            # Find which state to update
            update_s = np.random.randint(self.N_vis+self.N_hid)
            
            z = self.bias + self.W@state
            prob_active = self.logistic(z,T)
            
            if np.random.rand(1) > prob_active[update_s]:
                state[update_s] = 0
            else:
                state[update_s] = 1
                    
                
            states.append(state)


        self.state = state
        
        #print(state)
        
        if epochs == 1:
            states = self.state

        return states
    
    # Use decorator here??
    def iterate_clamped_vis(self, clamped_state, T = 1):
        
        state = self.state
        state[0 : self.N_vis] = clamped_state
        
        update_s = np.random.randint(self.N_hid)
        
        z = self.bias + self.W@state
        prob_active = self.logistic(z,T)

        hidden_index = update_s+self.N_vis
        if np.random.rand(1) > prob_active[hidden_index]:
            state[hidden_index] = 0
        else:
            state[hidden_index] = 1
        
        self.state = state
        
        return state
        
        
    def learn(self, data, it = 1000, alphaW = 0.03, alphab = 0.03):
        "Implement: Different step-sizes for each weight"
        # it = no. of iterations (iterates it times over all data)
        # Data have to be a 2D list, where a (first entry of list) row is one state vector of the visible units
        # alpha: learning rate
        from joblib import Parallel, delayed
        import multiprocessing
        
        print("\nGetting ready to do some learnin'\n")
        
        if not isinstance(data, list):
            data = data.tolist()
        
        num_cores = multiprocessing.cpu_count()
        print("Computing expectation values in data...")
        #expect_sisj_data, expect_si_data = self.expect_sisj_data(data)
        
        "==Parallelize NEEDS TO BE CHECKED=="
        length_data = len(data)
        data_parallel = [[]]*num_cores
        
        stepsize = int(np.floor(length_data / num_cores))
        for ip in range(num_cores):
            if ip < num_cores - 1: 
                data_parallel[ip] = data[ip*stepsize:(ip+1)*stepsize]
            else: 
                data_parallel[ip] = data[ip*stepsize:]
                restlength = len(data[ip*stepsize:])
        
        self.data_parallel = data_parallel
        
        print("\nnum_cores:%d"%num_cores)
        results = Parallel(n_jobs=num_cores)(delayed(self.expect_sisj_data_func)(data_parallel[i]) for i in range(num_cores))
        
        # print("\nresults:")
        # print(results)
        self.results = results
        
        expect_sisj_data, expect_si_data = 0,0
        for r in range(len(results)):
            if r < len(results) - 1 :
                expect_sisj_data += results[r][0]*stepsize
                expect_si_data += results[r][1]*stepsize
            elif r == len(results) - 1 :
                expect_sisj_data += results[r][0]*restlength
                expect_si_data += results[r][1]*restlength
        
        expect_sisj_data = expect_sisj_data/ length_data
        expect_si_data = expect_si_data/ length_data
        
        # print("\nexpect_sisj_data:\n")
        # print(expect_sisj_data)
        # print("\nexpect_si_data:\n")
        # print(expect_si_data)
        
        self.expect_sisj_data = expect_sisj_data
        self.expect_si_data = expect_si_data
        
        #diff_sisj = []
        #diff_si = []
        
        """Below code computes <s_i,s_j> and <s_i> of the model, then calculates
        the differences to <s_i,_j> and <_i> in the data and updates W and bias accordingly."""
        for i in range(it):
            print("\nStarting learning iteration %d of %d."%(i,it))
                
            expect_sisj_model, expect_si_model = self.expect_sisj_model_func()
            
            
            self.W = self.W + alphaW*(expect_sisj_data - expect_sisj_model)
            self.bias = self.bias + alphab*(expect_si_data - expect_si_model)
            
            if self.restricted:
                self.W[0:self.N_vis, 0:self.N_vis] = 0
                self.W[self.N_vis:, self.N_vis:] = 0

            #diff_sisj.append((expect_sisj_data - expect_sisj_model).sum())
            #diff_si.append((expect_si_data - expect_si_model).sum())
            
        #return diff_sisj, diff_si
        print("\nLearning Finished.")
        
    def expect_sisj_data_func(self, data):
        """data has to be a list of dimensions no_data_patterns*no_units (visible units)"""
        
        import time 
        from tqdm import tqdm
        start_time = time.time()
        
        ## Compute <s_i s_j> and <s_i> in the data
        expect_sisj_data = np.zeros((self.N_vis+self.N_hid,self.N_vis+\
                                     self.N_hid))
        
        expect_si_data = np.zeros(self.N_vis+self.N_hid)
        
        for idx, state in enumerate(data):
            print("\nInspecting data sample %d of %d.\n"%(idx+1, len(data)))
            t = idx+1
            
            if self.N_hid == 0:
                for i in range(self.N_vis):
                    expect_sisj_data[i,:] = expect_sisj_data[i,:]*(t-1)/t + state[i]*state/t

            else:
                hidits = 100
                print("\nPerforming %d iterations over hidden neurons.\n"%hidits)
                # Average over hidden state iterations
                for ii in tqdm(range(hidits)):
                    tt = ii + 1
                    itstate = self.iterate_clamped_vis(clamped_state = state[0:self.N_vis], T = 1)
                    
                    for i in range(self.N_vis+self.N_hid):
                        expect_sisj_data[i,:] = expect_sisj_data[i,:]*(tt-1)/tt + itstate[i]*itstate/tt
        
        
        
        expect_si_data = np.diag(expect_sisj_data)
        np.fill_diagonal(expect_sisj_data, 0)
        
        elapsed_time = time.time() - start_time
        print("\nElapsed time for data iteration is %.1f seconds. \n"%elapsed_time)
        return expect_sisj_data, expect_si_data
        
    def expect_sisj_model_func(self, iterations = 1000):
        """"Should have a lot of iterations as otherwise underestimates the probability
        of low-probability outcomes to be zero"""
        """Checked for consistency. Works, but only for N_hid == 0 (June 4th 2020)."""
        ## Compute <s_i s_j> and <si> in the model
        import time 
        from tqdm import tqdm
        start_time = time.time()
        
        expect_sisj_model = np.zeros((self.N_vis+self.N_hid, self.N_vis+self.N_hid))
        expect_si_model = np.zeros(self.N_vis+self.N_hid)
        print("\nLetting model run for %d iterations.\n"%iterations)
        #states = self.iterate(epochs = 10_000)
        
        for it in tqdm(range(iterations)):
            newstate = self.iterate()
            
            #newstate = states[it]
            t = it+1

            """Compute <s_i> (diagonal) and <s_is_j> (off-diagonal)"""
            for i in range(self.N_vis+self.N_hid):
                expect_sisj_model[i,:] = expect_sisj_model[i,:]*(t-1)/t + newstate[i]*newstate/t

        expect_si_model = np.diag(expect_sisj_model)
        np.fill_diagonal(expect_sisj_model, 0)
        
        elapsed_time = time.time() - start_time
        print("Elapsed time for model iteration with %d steps is %.1f seconds. \n"%(iterations,elapsed_time))
        return expect_sisj_model, expect_si_model
    
class batchBoltzmannM(BoltzmannM):
    
    def iterate(self, epochs = 1, T = 1):
        """output
        states: last network state after iteration (in case of 1 epoch).
                history of states during iteration (in case of epochs > 1)
        """
        #from tqdm import tqdm
        # Runs the generative model to create states
        # If BM did not run before, it will need some time to settle into a
        # stable state distribution
        # 1 epoch corresponds to 1 iteration step where all units get updated once        
        state = self.state

        states = []
        for e in range(epochs):
            z = self.bias + self.W@state
            prob_active = self.logistic(z,T)
            
            for i in range(self.N_vis + self.N_hid):
                if np.random.rand(1) > prob_active[i]:
                    state[i] = 0
                else:
                    state[i] = 1
                    
            states.append(state)
    
    
        self.state = state
        
        if epochs == 1:
            states = self.state

        return states
    
    # Use decorator here??
    def iterate_clamped_vis(self, clamped_state, T = 1):
        
        state = self.state
        state[0 : self.N_vis] = clamped_state
        
        z = self.bias + self.W@state
        prob_active = self.logistic(z,T)
        
        for i in range(self.N_hid):
            hidden_index = i+self.N_vis
            if np.random.rand(1) > prob_active[hidden_index]:
                state[hidden_index] = 0
            else:
                state[hidden_index] = 1
        
        self.state = state
        
        return state

class sequentialBM(batchBoltzmannM):
    """Has non-symmetrical connections"""
    
    