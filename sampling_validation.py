#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:24:05 2023

@author: s1984454
"""

from LoKi import LoKi
from LoKi_samp import LoKi_samp
import numpy as np
from scipy.spatial.distance import pdist

fname = 'dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0.3_epsilon_0.1_N_20000.txt'
dimensional_samples  = np.loadtxt(f'Data/{fname}')
G = 4.3009e-3

_, _, _, _, M,_,rK,_,Psi,_,mu,_,epsilon,_,_  = fname.split('_')
M, rK, Psi, mu, epsilon = [float(i) for i in [M, rK, Psi, mu, epsilon]]
N = len(dimensional_samples)

model = LoKi(mu, epsilon, Psi)
Mhat = model.M_hat
a = (9 * rK * Mhat)/(4*np.pi*G*M)
Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
A_hat = (8 * np.sqrt(2) * np.pi * Ae)/(3 * a**(3/2))
M_BH = A_hat * model.density(Psi) * rK**3 * mu

x = dimensional_samples[:, 0]
y = dimensional_samples[:, 1]
z = dimensional_samples[:, 2]

vx = dimensional_samples[:, 3]
vy = dimensional_samples[:, 4]
vz = dimensional_samples[:, 5]


m = M/N

 
K_i = 0.5 * m * (vx**2 + vy**2 + vz**2)
K = np.sum(K_i)

U = 0
for i in range(N-1):
    
    x1 = x[i]
    y1 = y[i]
    z1 = z[i]

    x2_array = x[i+1:]
    y2_array = y[i+1:]
    z2_array = z[i+1:]
    
    separations = np.sqrt((x1 - x2_array)**2 + (y1 - y2_array)**2 + (z1 - z2_array)**2)
    contribution_to_U = -G * m**2 / separations
    
    U = U + np.sum(contribution_to_U) 

print(-K/U)