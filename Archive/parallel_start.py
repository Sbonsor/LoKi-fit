#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:31:33 2024

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
from mpi4py import MPI
from scipy.optimize import minimize

def r_v(data):
    
    rs = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    vs = np.sqrt(data[:,3]**2 + data[:,4]**2 + data[:,5]**2)
    
    return rs, vs

def radii_separations(Psi, rK):
    
    model = LoKi(0, 1e-6, Psi)
    r2 = np.interp(0.8*model.M_hat, model.M_r,model.rhat) * rK
    rt = model.rhat[-1]*rK
    
    return [1e-6, rK, r2, rt]

def split_data_by_radius(data, region_boundaries):
    
    rs, vs = r_v(data)
    
    inner_indices = np.where(rs<=region_boundaries[1])[0]
    intermediate_indices = np.where(np.logical_and(rs>region_boundaries[1], rs<=region_boundaries[2]))[0]
    outer_indices = np.where(rs>region_boundaries[2])[0]
    
    return [data[inner_indices,:], data[intermediate_indices,:], data[outer_indices,:]]

def sum_single_star_log_likelihood(theta, rs, vs, r1, r2):
    
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
        ### Calculate dimensional scales.
        model = LoKi(mu, epsilon, Psi, pot_only = False)
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        a = (9 * rK * Mhat)/(4*np.pi*G*M)
        Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
        M_scale = M/Mhat
        
        ### Caclulate individual star likelihoods        
        rhats = rs/rK
        vhats = vs*np.sqrt(a)
                
        psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
        Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
            
        ls = (Ae)/M * (np.exp(-Ehats) - 1)
        
        log_ls = np.log(ls)
        
        return np.sum(log_ls)

M = 500
rK = 1.2
Psi = 5
theta0 = np.array([M, rK, Psi])
data_path = '/home/s1984454/Desktop/LoKi-Fit/Data/dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_20000.txt'
data = np.loadtxt(data_path)
N = len(data)
M_min = 450
M_max = 550
rK_min = 1 
rK_max = 1.4
Psi_min = 3
Psi_max = 7
parameter_bounds = [(M_min, M_max), (rK_min, rK_max), (Psi_min, Psi_max)] 
h = 1e-4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

region = 0
region_boundaries = radii_separations(Psi, rK)


if (rank == 0):
    
    data = np.loadtxt(data_path)
    split_data = split_data_by_radius(data, region_boundaries)
    region_data = np.array_split(split_data[region], size)
    
else:
    region_data = None
    
region_data = comm.scatter(region_data, root = 0) # Distribute each sub-array to it's processor.



