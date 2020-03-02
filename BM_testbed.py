#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:24:54 2020

@author: sascha
"""

import mnist
import scipy.misc
import matplotlib.pyplot as plt
import BM
import numpy as np

images = mnist.train_images()

data = np.reshape(images, (images.shape[0],images.shape[1]*images.shape[2]))

bm = BM.BoltzmannM(28*28)