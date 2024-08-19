#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:17:18 2024

@author: s1984454
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

def cov_cross_element(array1, array2):
    
    cov = np.cov(array1, array2, ddof = 0)
    
    return cov[0,1]

def construct_cov_array(matrix1, matrix2 ):
    
    cov_array = np.zeros(3)
    
    for i in range(3):
        cov_array[i] = cov_cross_element(matrix1[:,i], matrix2[:,i])
        
    return cov_array


###### Convergence calculation
base_path = 'Data/SAMPLES_MCMC_auto_proposal_M_500_rK_1.2_Psi_5_non_diag_'
indices = [0,1,2]
labels = ['M', 'rK', 'Psi']
m = 10

samples = np.loadtxt(f'{base_path}{0}.txt')
n = len(samples)//2
xi_bar = np.zeros((m,3))
si_2 = np.zeros((m,3))
total_sum = 0

for i in range(m):
    
    samples = np.loadtxt(f'{base_path}{i}.txt')
    samples = samples[n:, indices]
    
    xi_bar[i,:] = np.mean(samples, axis = 0)
    si_2[i,:] = np.var(samples, axis = 0)
    
    total_sum += np.sum(samples, axis = 0)

mu_hat = xdotdot = total_sum / (n*m)

squared_centered_means = (xi_bar - np.tile(xdotdot, (m,1)))**2

B_n = np.sum(squared_centered_means, axis = 0) / (m-1)
W = np.mean(si_2, axis = 0)

sigma_hat_2 = (n-1)/n * W + B_n

root_V_hat = np.sqrt(sigma_hat_2 + B_n/m) 

var_si = np.var(si_2, axis = 0)

cov1 = construct_cov_array(si_2, xi_bar**2)
cov2 = construct_cov_array(si_2, xi_bar)
var_V_hat = np.square((n-1)/n)/m * var_si + np.square((m+1)/(m*n))*((2*n**2)/(m-1))* B_n**2 + (2*(m+1)*(n-1))/(m**2*n) *(cov1 - 2 * xdotdot * cov2) 
df = 2 * root_V_hat**4/var_V_hat

root_Rhat = np.sqrt((root_V_hat**2 / W) * (df/(df-2)))

###### Corner plot
path_to_samples = 'Data/SAMPLES_MCMC_auto_proposal_M_500_rK_1.2_Psi_5_non_diag_3.txt'
burn_in = 50000
indices = [0,1,2]

stack_samp = np.zeros((m * n, 3))

for i in range(m):
    samples = np.loadtxt(f'{base_path}{i}.txt')
    stack_samp[i*n:(i+1)*n] = samples[burn_in:, indices]


fig = corner.corner(stack_samp, labels = ['$M$','$r_K$', '$\\Psi$'], quantiles = [0.5])







