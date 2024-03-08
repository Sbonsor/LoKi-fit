#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:24:03 2024

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import simps
from scipy.special import gammainc, gamma
from scipy.optimize import minimize
from mpi4py import MPI

def r_v(data):
    
    rs = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    vs = np.sqrt(data[:,3]**2 + data[:,4]**2 + data[:,5]**2)
    
    return rs, vs

def radii_separations(mu, Psi, rK):
    
    model = LoKi(mu, 1e-6, Psi)
    r1 = np.interp(0.5*model.M_hat, model.M_r,model.rhat) * rK
    r2 = np.interp(0.8*model.M_hat, model.M_r,model.rhat) * rK
    r3 = model.rhat[-1]*rK
    
    return r1, r2, r3

def split_data_by_radius(data, r1, r2):
    
    rs, vs = r_v(data)
    
    inner_indices = np.where(rs<=r1)[0]
    intermediate_indices = np.where(np.logical_and(rs>r1, rs<=r2))[0]
    outer_indices = np.where(rs>r2)[0]
    
    return data[inner_indices,:], data[intermediate_indices,:], data[outer_indices,:]

def log_likelihoods(theta, rs, vs):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    
    mu = 0
    epsilon = 1e-6
    G = 4.3009e-3
    
    a0 = Psi - (9*mu)/(4*np.pi)
    
    if(a0 <=0):
        return -np.inf
    else:
        model = LoKi(mu, epsilon, Psi, pot_only = True)
        
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        a = (9 * rK * Mhat)/(4*np.pi*G*M)
        Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
            
        rhats = rs/rK
        vhats = vs*np.sqrt(a)
                
        psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
        Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
            
        ls = (Ae)/M * (np.exp(-Ehats) - 1)
        
        log_ls= np.log(ls)
        
        return log_ls
    
def theta_star(max_Psi, min_Psi, max_rK, min_rK, max_M, min_M, n_grid, data):
    
    rs, vs = r_v(data)

    Psis = np.linspace(min_Psi, max_Psi, n_grid)
    Ms = np.linspace(min_M, max_M, n_grid)
    rKs = np.linspace(min_rK, max_rK, n_grid)

    PSIS, MS, RKS = np.meshgrid(Psis, Ms, rKs)

    PSIS = np.ndarray.flatten(PSIS)
    MS = np.ndarray.flatten(MS)
    RKS = np.ndarray.flatten(RKS)

    result = []

    for i in range(len(PSIS)):
        theta = np.array([MS[i], RKS[i], PSIS[i]])
        log_ls = log_likelihoods(theta, rs, vs)
        result.append(np.sum(log_ls))
        
    max_idx = np.where(log_likelihoods == np.max(log_likelihoods))[0][0]

    theta_star = [MS[max_idx], RKS[max_idx], PSIS[max_idx]]
    
    return theta_star

def single_derivative_log_l(i, h, rs, vs, theta):
    
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
 
    u_x2 = log_likelihoods(theta + 2*x_increment, rs, vs)
    u_x1 = log_likelihoods(theta + 1*x_increment, rs, vs)
    u_1x = log_likelihoods(theta - 1*x_increment, rs, vs)
    u_2x = log_likelihoods(theta - 2*x_increment, rs, vs)
    
    first_deriv = ((u_x2/12) - (2*u_x1/3) + (2*u_1x/3) - (u_2x/12))/h
    
    return first_deriv

def mass_diff_first_derivative(i, h, theta, region_boundary):
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
     
    u_x2 = mass_in_region(theta + 2*x_increment, region_boundary)
    u_x1 = mass_in_region(theta + 1*x_increment, region_boundary)
    u_1x = mass_in_region(theta - 1*x_increment, region_boundary)
    u_2x = mass_in_region(theta - 2*x_increment, region_boundary)
    
    first_deriv = ((u_x2/12) - (2*u_x1/3) + (2*u_1x/3) - (u_2x/12))/h
    
    return first_deriv

def mass_in_region(theta, region_boundary):
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    
    model = LoKi(mu, epsilon, Psi)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    M_scale = M/Mhat
    Mass_profile = model.M_r *M_scale
    r = model.rhat * rK
    
    mass_diff = np.interp(region_boundary[1], xp = r, fp = Mass_profile) - np.interp(region_boundary[0], xp = r, fp = Mass_profile) 
    
    return mass_diff

### True model paramters
M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6
theta = np.array([M, rK, Psi])

### File containing the star samples
data_file = 'dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000.txt'

### Step size for numerical differentiation
h = 1e-4

### Parameters for maximising likelihood
n_grid = 10
min_Psi = 4.5
max_Psi = 5.5
min_M = 490
max_M = 510
min_rK = 1.1
max_rK = 1.3

###

### Split data into the three regions of interest.
data = np.loadtxt(f'Data/{data_file}')

half_mass_r, eighty_perc_r, rt = radii_separations(mu, Psi, rK)

region_boundaries = [(epsilon, rK), (rK, eighty_perc_r), (eighty_perc_r, rt)]

split_data = split_data_by_radius(data, rK, eighty_perc_r)


region = 0
### Calculate maximum likelihood theta. (for now just using inner data as test case)
region_data = split_data[region]

theta_star = theta_star(max_Psi, min_Psi, max_rK, min_rK, max_M, min_M, n_grid, region_data)

### Sample parameters
N = len(data)
m = M/N
n = len(region_data)

### Calculating sample derivatives of log likelihood
rs, vs = r_v(region_data)

derivatives_log_l = [np.sum(single_derivative_log_l(i, h, rs, vs, theta_star)) for i in range(len(theta_star))] 

### Calculating derivative of mass function
derivatives_mass_diff = [(1/m)*mass_diff_first_derivative(i, h, theta_star, region_boundaries[region]) for i in range(len(theta_star))]

information_matrix = np.zeros((3,3))

for i in range(3):
    for j in range(i,3):
        information_matrix[i,j] = (derivatives_log_l[i] + derivatives_mass_diff[i]) * (derivatives_log_l[j] + derivatives_mass_diff[j])










