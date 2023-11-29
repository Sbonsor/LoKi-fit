#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:33:52 2023

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import gammainc, gamma
import emcee

def l_prior(self, parameters):
    
    Psi, a0, M, rK = parameters
    Psi_max, Psi_min, M_max, M_min, rK_max, rK_min = self.prior_args
    
    Psi_flag = (Psi >= Psi_min) and (Psi <= Psi_max)
    a0_flag = (a0 >= 0) and (a0 <= Psi)
    M_flag = (M >= M_min) and (M <= M_max)
    rK_flag = (rK >= rK_min) and (rK <= rK_max)
    
    if (Psi_flag and a0_flag and M_flag and rK_flag):
        
        V = 0.5 * (rK_max - rK_min) * (M_max - M_min) * (Psi_max**2 - Psi_min**2)
        log_prior = np.log(1/V)
        
    else:
        
        log_prior = -np.inf
        
    return log_prior

def log_likelihood_6d(theta, x, y, z, vx, vy, vz):
    
    Psi, a0, M, rK = theta
    
    mu = (Psi - a0) * 4*np.pi*epsilon/9
    
    ##calculate constant factors in DF definition.
    model = LoKi(mu, epsilon, Psi, pot_only = True)
    M_hat = -mu - 4*np.pi * model.rhat[-1]**2 * model.dpsi_dr[-1]/9
    a = (9 * rK * M_hat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * M_hat)
    
    xs = x.copy()
    ys = y.copy()
    zs = z.copy()
    vxs = vx.copy()
    vys = vy.copy()
    vzs = vz.copy() 
    
    xs *= 1/rK
    ys *= 1/rK
    zs *= 1/rK
    vxs *= np.sqrt(a)
    vys *= np.sqrt(a)
    vzs *= np.sqrt(a)
    
    rhats = np.sqrt(x**2 + y**2 + z**2)
    vhats = np.sqrt(vx**2 + vy**2 + vz**2)
    
    psi = np.interp(rhats, xp = model.rhat, fp = model.psi)
    Ehats = np.clip(0.5* vhats**2 - psi, a_max = 0, a_min = None)
    
    likelihoods = (Ae)/M * (np.exp(-Ehats) - 1)
    
    ### Return -infinity if there is a zero likelihood for a point. Otherwise take the logarithm and sum.
    if (0 in likelihoods):
        
        return -np.inf
    
    else:
        
        l_p = l_prior(theta) 
        
        if not np.isfinite(l_p):
            
            return -np.inf
        
        else:
            
            return np.sum(np.log(likelihoods)) + l_p
  
G = 4.3009e-3
epsilon = 0.1
    
M_true = 500
mu_true = 0.3
Psi_true = 5
a0_true = Psi_true - (9*mu_true)/(4*np.pi*epsilon)
rK_true = 1.2

nwalkers = 10
ndim = 4
nsamp = 5000

#data_path = '/home/s1984454/LoKi-fit/Data/'
data_path = '/home/s1984454/Desktop/LoKi-Fit/Data/'
fname = f'dimensional_samples_King_M_{M_true}_rK_{rK_true}_Psi_{Psi_true}_mu_{mu_true}_epsilon_{epsilon}_N_20000'
data = np.loadtxt(data_path + fname + '.txt') 

x = data[:,0]
y = data[:,1]
z = data[:,2]
vx = data[:,3]
vy = data[:,4]
vz = data[:,5]

pos = np.array([Psi_true, a0_true, M_true, rK_true]) + 1e-2 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_6d, args = (x,y,z,vx,vy,vz) )

sampler.run_mcmc(pos, nsamp, progress = True)

samples = sampler.get_chain()

labels = ['$\\Psi$', '$a_0$', '$M$', '$r_K$']
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");









