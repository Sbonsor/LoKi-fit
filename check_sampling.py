#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:33:33 2023

@author: s1984454
"""

import numpy as np
from scipy.special import gammainc,gamma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from LoKi import LoKi
from scipy.spatial.distance import pdist
from scipy.optimize import root_scalar
from dimensional_data_generation import dimensional_data_generation

# def density_normalisation(r,rho):
#     integrand = r**2*rho
#     return np.trapz(y = integrand, x = r)   

# def cdf_v(v, max_v, psi):
    
#     numerator =   np.sqrt(2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,v**2/2) - v**3/3
#     denominator = np.sqrt(2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,max_v**2/2) - max_v**3/3
    
#     return numerator/denominator

# def sample_and_verify(Psi, mu, epsilon, M, rK, N):
    
#     G = 4.3009e-3

#     model = LoKi(mu, epsilon, Psi)    
            
#     rand1 = np.random.rand(N)

#     r = np.interp(rand1, model.M_r/model.M_hat, model.rhat)

#     rand2 = np.random.rand(N)
#     rand3 = np.random.rand(N)

#     r2 = r ** 2
#     z = (1-2*rand2) * r
#     x = np.sqrt(r2 - z**2)*np.cos(2*np.pi*rand3)
#     y = np.sqrt(r2 - z**2)*np.sin(2*np.pi*rand3)

#     psi_samples = np.interp(r,model.rhat,model.psi)

#     v_samples = np.zeros(N)

#     for j in range(N):

#         psi_j = psi_samples[j]
#         r_j = r[j]
        
#         v_max = np.sqrt(2*psi_j)
#         vs = np.linspace(0,v_max,10000)
#         cdf = cdf_v(vs, v_max, psi_j)
    
#         rand4 = np.random.rand()
#         v_samples[j] = np.interp(rand4, cdf, vs)

#     rand6 = np.random.rand(N)
#     rand7 = np.random.rand(N)

#     v2 = v_samples ** 2
#     vz = (1-2*rand6) * v_samples
#     vx = np.sqrt(v2 - vz**2)*np.cos(2*np.pi*rand7)
#     vy = np.sqrt(v2 - vz**2)*np.sin(2*np.pi*rand7)

#     A_hat = M/(model.M_hat * model.density(Psi) * rK**3)
#     sqrt_a = np.sqrt( 9/(4 * np.pi * G * A_hat * model.density(Psi) * rK**2) )

#     x = x * rK
#     y = y * rK
#     z = z * rK

#     vx = vx/sqrt_a
#     vy = vy/sqrt_a
#     vz = vz/sqrt_a
    
#     Mhat = model.M_hat
#     a = (9 * rK * Mhat)/(4*np.pi*G*M)
#     Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
#     A_hat = (8 * np.sqrt(2) * np.pi * Ae)/(3 * a**(3/2))
#     M_BH = A_hat * model.density(Psi) * rK**3 * mu
#     m = M/N

#     K_i = 0.5 * m * (vx**2 + vy**2 + vz**2)
#     K = np.sum(K_i)

#     U = 0
#     for i in range(N-1):
        
#         x1 = x[i]
#         y1 = y[i]
#         z1 = z[i]

#         x2_array = x[i+1:]
#         y2_array = y[i+1:]
#         z2_array = z[i+1:]
        
#         separations = np.sqrt((x1 - x2_array)**2 + (y1 - y2_array)**2 + (z1 - z2_array)**2)
#         contribution_to_U = -G * m**2 / separations
        
#         U = U + np.sum(contribution_to_U)
    
#     radii = np.sqrt(x**2 + y**2 + z**2)
#     U_BH = np.sum(-G*m*M_BH/radii)
    
#     U = U + U_BH
#         #print(i)
        
#     return -K/U

N_values = np.logspace(np.log10(1000), np.log10(100000), 6, dtype = int)
n_reps = 10
Psi = 5
mu = 0.3
epsilon = 0.1
M = 500
rK = 1.2

Ns = []
for N in N_values:
    for i in range(n_reps):
        Ns.append(N)        

virial_ratios = np.zeros(len(Ns))

for i, N in enumerate(Ns):
    
    sampling = dimensional_data_generation(N, M, rK, Psi, mu, epsilon, save = False, validate = True)
    virial_ratios[i] = sampling.virial_ratio
    
    print(N)

deviation_in_virial_ratio = virial_ratios - 0.5

fig1,ax1 = plt.subplots(1,1)
ax1.scatter(Ns, virial_ratios, marker = 'x')
ax1.set_xlabel('N')
ax1.set_ylabel('$Q_{vir}$')
ax1.set_xscale('log')
ax1.axhline(y = 0.5, linestyle = '--', color = 'k')

fig2,ax2 = plt.subplots(1,1)
ax2.scatter(Ns, abs(deviation_in_virial_ratio), marker = 'x')
ax2.set_xlabel('N')
ax2.set_ylabel('$Q_{vir}$ deviation')
ax2.set_xscale('log')
ax2.set_yscale('log')























