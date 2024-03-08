#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:37:19 2024

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
    mu = 0
    epsilon = 1e-6
    
    model = LoKi(mu, epsilon, Psi)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    M_scale = M/Mhat
    Mass_profile = model.M_r *M_scale
    r = model.rhat * rK
    
    mass_diff = np.interp(region_boundary[1], xp = r, fp = Mass_profile) - np.interp(region_boundary[0], xp = r, fp = Mass_profile) 
    
    return mass_diff

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
 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()    
 
### True model paramters
M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6
theta = np.array([M, rK, Psi])

### File containing the star samples
data_path = '/home/s1984454/LoKi-fit/Data/dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000.txt'

### Step size for numerical differentiation
h = 1e-4

for region in range(3):

    if (rank == 0):
        print(f'Calculating through region {region}...')
        ###Read in full data set
        data = np.loadtxt(data_path) #Load data
        
        ### Split data into the three regions of interest.
        half_mass_r, eighty_perc_r, rt = radii_separations(mu, Psi, rK)
        split_data = split_data_by_radius(data, rK, eighty_perc_r)
        data = np.array_split(split_data[region], size) #Generate list with subsets of the data to go to each processor
        
    else:
        
        data = None
        
    data = comm.scatter(data,root = 0) # Distribute each sub-array to it's processor.
    
    theta_star = np.loadtxt(f'max_likelihood_params_region_{region}.txt') #load the previously calculated maximum likelihood theta for this region
    theta_star = theta_star[0:3]
    
    rs, vs = r_v(data)

    derivatives_log_l = [np.sum(single_derivative_log_l(i, h, rs, vs, theta_star)) for i in range(len(theta_star))] # Calculate the sum of the first derivatives for this processor's data points
    
    if (rank == 0):
        recv_buff = np.empty([size, 3], dtype = float)

    comm.Gather(derivatives_log_l, recv_buff, root = 0)
    
    if(rank == 0):
        array_to_save = sum(recv_buff)
        save_file = '/home/s1984454/LoKi-fit/Data/log_likelihood_derivatives.txt'
        with open(save_file, 'ab') as f:
            np.savetxt(f, array_to_save, delimiter= ' ')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    