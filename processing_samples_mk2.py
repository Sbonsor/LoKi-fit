#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:09:58 2024

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import simps
from scipy.special import gammainc, gamma
import time

path_to_samples = 'Data/SAMPLES_run_13_0.txt'
burn_in = 25000
indices = [0,1,2,3]
samples = np.loadtxt(path_to_samples)
stack_samp = samples[:, indices]
fig = corner.corner(stack_samp, labels = ['$M$','$r_K$', '$\\mu$', '$\\Psi$'], quantiles = [0.5])

#### 1 fixed, BH containing
path_to_samples = 'Data/SAMPLES_run_13_0.txt'
burn_in = 25000
indices = [0,1,2,3]
samples = np.loadtxt(path_to_samples)
stack_samp = samples[:, indices]
fig = corner.corner(stack_samp, labels = ['$M$','$r_K$', '$\\mu$', '$\\Psi$'], quantiles = [0.5])






