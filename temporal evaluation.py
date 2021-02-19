#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:34:55 2020

@author: sascha
"""

import BM

no_iters = 10
bm = BM.BoltzmannM(10, init_state=np.zeros(10), T = 0.01)
state, state_hist = bm.iterate(no_iters, save_history = True)


#%%
print("Starting temporal evaluation")

# Go through all different states in states_hist and compute their 
# Boltzmann-prob as exp(-E(state))/const.
# Remove all duplicates from states_hist
new_array = [np.array(row) for row in state_hist]
uniques = np.unique(new_array, axis=0)

# For all states in uniques, compute energy
E_uniques = []
for s in uniques:
    E_uniques.append(bm.energy(s))

# Create dict with idx of unique states as keys and energies as values
d = {}
for idx in range(len(E_uniques)):
    d[idx] = E_uniques[idx]

# Sort array according to energy values
d_sorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

# Go through uniques and determine their states relative occurrence in 
# state_hist and how often they should have occurred according to the 
# Boltzmann distro
rel_occ_in_uniques = []
boltzmann_occ = []

print("During the iteration of %d iterations, %d different states were visited."\
      %(no_iters, uniques.shape[0]))

for idx in tqdm(range(uniques.shape[0])):
    occurrences = 0
    for jdx in range(len(state_hist)):
        if sum(uniques[idx] == state_hist[jdx]) == 10:
            occurrences += 1

    rel_occ_in_uniques.append(occurrences/no_iters)
    boltzmann_occ.append(np.exp(-bm.energy(uniques[idx])))

# boltzmann occurrences should ad up to 1 as they are probabilities
# boltzmann_occurrences[i] equals the probability with which state vector
# i should occur over time.
boltzmann_occ = boltzmann_occ/sum(boltzmann_occ)

#return boltzmann_occ, rel_occ_in_uniques, E_hist