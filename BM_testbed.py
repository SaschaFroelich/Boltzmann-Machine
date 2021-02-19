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
from PIL import Image
import random

mndata = MNIST('python-mnist/data')
images, labels = mndata.load_training()
 

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


#%% Train BM on digit 8
random.seed(111)

disrupted_eight = Image.open('images/disrupted_8_inpt.png').convert('L')
disrupted_eight = np.array(disrupted_eight).reshape((28*28,1))
disr_8_inpt = [0 if i<=30 else 1 for i in disrupted_eight]

data = []
for i in range(len(images_bin)):
    if labels[i] == 8:
        data.append(images_bin[i])

"""Initialise Boltzmann Machine"""
bm = BM.BoltzmannM(28*28, no_hid=28*28)
bm.state[0:28*28] = disr_8_inpt
state = bm.iterate(T=0.001)
imvec = np.reshape(state[0:28*28], (28,28)).tolist()
i=Image.fromarray(np.array(imvec, dtype=np.uint8)*200,"L")
name = "images/before_disr8.png"
i = i.resize((200,200))
i.save(name)

bm.learn(data, it = 10_000)

#%%
"Create images before learning Before learning"

for idx in range(10):
    #state = bm.iterate(T=0.001)
    #state = state[0:28*28]
    state = images_bin[17]
    imvec = np.reshape(state, (28,28)).tolist()
    i=Image.fromarray(np.array(imvec, dtype=np.uint8)*200,"L")
    name = "images/before_image{0}.png".format(idx)
    i = i.resize((200,200))
    i.save(name)

#%%

bm.learn(data, it = 600)

#%%

#bm.learn(data, it = 600)
bm.learn(images_bin, it = 600)

# Iterate BM so it settles into stable distribution 

_ = bm.iterate(10_000)

#%%
testdata = [[1,1,1],[1,0,1],[0,0,1],[1,1,1]]
expect_sisj_model = np.zeros((3,3))

for it in range(len(testdata)):
    t=it+1
    newstate = np.array(testdata[it])
    for i in range(3):
        #breakpoint()
        expect_sisj_model[i,:] = expect_sisj_model[i,:]*(t-1)/t + newstate[i]*newstate/t

#%% Iterate a number of random states.
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