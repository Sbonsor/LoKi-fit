#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:34:39 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np

M = 500
rK = 1.2
Psi = 5
N = 1000000
n = 10000

region = 0

matrix_file = f'/home/s1984454/Desktop/LoKi-Fit/Data/M_{M}_rK_{rK}_Psi_{Psi}_N_{N}_n_{n}_region_{region}_information_matrix.txt'

I = np.loadtxt(matrix_file)

# for region in range(3):

#     matrix_file = f'/home/s1984454/Desktop/LoKi-Fit/Data/M_{M}_rK_{rK}_Psi_{Psi}_N_{N}_n_{n}_region_{region}_information_matrix.txt'

#     I = np.loadtxt(matrix_file)
    
#     print(np.linalg.det(I))

def log_likelihood(theta):
    
    return theta**2

def single_derivative_log_l(i, h, theta):
    
    #x_increment = np.zeros(len(theta))
    x_increment = h
 
    u_x2 = log_likelihood(theta + 2*x_increment)
    u_x1 = log_likelihood(theta + 1*x_increment)
    u_1x = log_likelihood(theta - 1*x_increment)
    u_2x = log_likelihood(theta - 2*x_increment)
    
    first_deriv = (-(u_x2/12) + (2*u_x1/3) - (2*u_1x/3) + (u_2x/12))/h
    
    return first_deriv

print(single_derivative_log_l(0,1e-4, 5))