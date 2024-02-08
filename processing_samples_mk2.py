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

def means_covariance(samples):

    n_parameters = np.shape(samples)[1]     
    
    means = np.zeros(n_parameters)
    cov_matrix = np.zeros((n_parameters,n_parameters))
    
    for i in range(n_parameters):
        
        means[i] = np.mean(samples[:,i])
        i_centered = samples[:,i] - np.mean(samples[:,i])
        
        for j in range(n_parameters):
            
            j_centered = samples[:,j] - np.mean(samples[:,j])
            
            cov_matrix[i][j] = np.mean(i_centered*j_centered)
    
    return means, cov_matrix

def process_samples(path_to_samples, indices, burn_in):
    
    labels = ['$\\rho_0$', '$r_k$', '$M_{BH}$', '$\\Psi$', '$\\epsilon$']
    
    samples = np.loadtxt(path_to_samples)
    stack_samp = samples[burn_in:, indices]
    
    mean, covariance = means_covariance(stack_samp)
    
    fig = corner.corner(stack_samp, labels = [labels[indices[0]], labels[indices[1]], labels[indices[2]]], quantiles = [0.5], hist_args=dict(density=True))
    
    axes = np.array(fig.axes).reshape((3, 3))
    
    # for i in range(3):
    #     mu = mean[i]
    #     sig = np.sqrt(covariance[i,i])
    #     x = np.linspace(min(stack_samp[:,i]), max(stack_samp[:,i]), 100)
    #     y = gaussian(x, mu, sig)
        
    #     fig, ax = plt.subplots(1,1)
    #     ax.hist(stack_samp[:,i], density = True, bins = 20)
    #     ax.plot(x,y, color = 'r')
    
    return mean, covariance

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

# path_to_samples = 'Data/SAMPLES_run_3.txt'
# burn_in = 25000
# indices = [0,1,3]

# mean1, covariance1 = process_samples(path_to_samples, indices, burn_in)

# path_to_samples = 'Data/SAMPLES_run_4.txt'
# burn_in = 25000
# indices = [0,1,3]

# mean2, covariance2 = process_samples(path_to_samples, indices, burn_in)

path_to_samples = 'Data/SAMPLES_run_7_4.txt'
burn_in = 50000
indices = [0,1,3]

mean3, covariance3 = process_samples(path_to_samples, indices, burn_in)





