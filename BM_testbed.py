#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:24:54 2020

@author: sascha
"""

from mnist import MNIST
import scipy.misc
import matplotlib.pyplot as plt
import BM
import numpy as np
from pylab import *
import operator as op
from tqdm import tqdm
mndata = MNIST('python-mnist/data')
images, labels = mndata.load_training()
 
N = 5
bmtest = BM.BoltzmannM(N)

#%%
def ncr(n, r):
    # nchooser(n,r)
    from functools import reduce
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


#%%
"""Train with binarised MNIST dataset"""
#images = mnist.train_images()
#labels = mnist.train_labels()

"""Only train on numbers 1, 5 and 8 for simplicity"""
#rem = []
#for idx, label in enumerate(labels):
#    if label == 1 or label == 5 or label == 8:
#        pass
#    else:
#        rem += [idx]
    
"""Binarise images"""
images_bin = images.copy()
#images_bin[images_bin <= 30] = 0
#images_bin[images_bin > 30] = 1

images_bin=images_bin[:1000]
for row in tqdm(range(len(images_bin))):
    digit = [0 if i<=30 else 1 for i in images_bin[row]]
    images_bin[row] = digit

del images

#data = np.reshape(images_bin, (images_bin.shape[0],images_bin.shape[1]*images_bin.shape[2]))

#print("len of rem is %d"%len(rem))

#for i in reversed(rem):
#    print("Removing %d"%i)
#    data = np.delete(data, i, axis = 0)
    
    
#del rem
#del images_bin


"""Initialise Boltzmann Machine"""
bm = BM.BoltzmannM(28*28)

#%%

#bm.learn(data, it = 600)
bm.learn(images_bin, it = 600)

# Iterate BM so it settles into stable distribution 

_ = bm.iterate(10_000)

#%%
statehist = bm.iterate(100, savehist=True)

from PIL import Image

idx = 0
for state in statehist:
    idx+=1
    imvec = np.reshape(state, (28,28)).tolist()
    i=Image.fromarray(np.array(imvec, dtype=np.uint8)*200,"L")
    name = "images/image{0}.png".format(idx)
    i = i.resize((200,200))
    # i.show()
    i.save(name)
#%%

"""End of MNIST"""
#%%
"""Save Object"""
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(bm, 'images/bm_more_learning.pkl')
#%%
with open('bm.pkl', 'rb') as input:
    bla = pickle.load(input)

#%%
# Crate all 1024 random state vectors for N neurons
import itertools
N = 5
testdata = [list(i) for i in itertools.product([0, 1], repeat=N)]
# Assign them all a probability to occur in the real world.
# Sample from fat-tailed Poisson distro
probs = np.random.poisson(lam=2, size=len(testdata))
probs = probs/sum(probs)

bmtest = BM.BoltzmannM(N)


#%%
total_no_states = 0
N = 100
for n in range(N+1):
    #n is the no. of "on" states
    total_no_states += ncr(N,n)
    
print("There is a total of %d different possible states for %d neurons"\
      %(total_no_states,N))

#%%
bmtest.expect_sisj_model(iterations = 10)