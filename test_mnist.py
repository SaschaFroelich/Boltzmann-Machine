#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:01:37 2020

@author: sascha
"""

import BM
from mnist import MNIST
import numpy as np
from PIL import Image  

mndata = MNIST('/home/sascha/Desktop/Boltzmann-Machine/')
images, labels = mndata.load_training()

# Only use labels 3, 5, and 7
data = [images[i] for i in range(len(labels)) if labels[i] == 3 or labels[i] == 5 or labels[i] == 7]

bm = BM.BoltzmannM(28*28)
bm.learn(data)

"""Generate random vectors"""
bm.learn(it=100_000)

statehist = bm.iterate(1000, savehist=True)

for i, state in enumerate(statehist):
    state[state == 1] = 253
    im = Image.fromarray(np.reshape(state,(28,28)).astype(np.uint8))
    im.save('im%d.png'%i)