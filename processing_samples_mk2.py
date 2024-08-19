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
plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 16

# path_to_samples = 'Data/SAMPLES_MCMC_auto_proposal_M_500_rK_1.2_Psi_5_0.txt'
# burn_in = 25000
# indices = [0,1,3]
# samples = np.loadtxt(path_to_samples)
# stack_samp = samples[:, indices]
# fig = corner.corner(stack_samp, labels = ['$M$','$r_K$', '$\\Psi$'], quantiles = [0.5])

#### 0 fixed, BH containing
# path_to_samples = 'Data/SAMPLES_run_13_0.txt'
# burn_in = 25000
# indices = [0,1,2,3,4]
# samples = np.loadtxt(path_to_samples)
# stack_samp = samples[:, indices]
# fig = corner.corner(stack_samp, labels = ['$M$','$r_K$','$\\mu$', '$\\Psi$', '$\\epsilon$'], quantiles = [0.5])

# path_to_samples = 'Data/SAMPLES_incomplete_data_0.txt'
# burn_in = 25000
# indices = [0,1,2]
# samples = np.loadtxt(path_to_samples)
# stack_samp = samples[:, indices]
# fig = corner.corner(stack_samp, labels = ['$M$','$r_K$', '$\\Psi$'], quantiles = [0.5])

path_to_samples = 'Data/SAMPLES_MCMC_auto_proposal_M_500_rK_1.2_Psi_5_non_diag_3.txt'
burn_in = 50000
indices = [0,1,2]
samples = np.loadtxt(path_to_samples)
stack_samp = samples[burn_in:, indices]
fig = corner.corner(stack_samp, labels = ['$M$','$r_K$', '$\\Psi$'], quantiles = [0.5])