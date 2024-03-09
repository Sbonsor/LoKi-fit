#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:46:02 2024

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
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
        
        for l in ls:
            log_ls= np.log(ls)
        
        return log_ls

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6
theta = np.array([M, rK, Psi])

### File containing the star samples
data_path = '/home/s1984454/LoKi-fit/Data/dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000.txt'
#data_path = '/home/s1984454/Desktop/LoKi-Fit/Data/dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000.txt'
### Parameters for maximising likelihood
n_grid = 100
min_Psi = 4.5
max_Psi = 5.5
min_M = 490
max_M = 510
min_rK = 1.1
max_rK = 1.3

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
    
    rs, vs = r_v(data)
    ### Create a list of all triplets on the grid.
    Psis = np.linspace(min_Psi, max_Psi, n_grid)
    Ms = np.linspace(min_M, max_M, n_grid)
    rKs = np.linspace(min_rK, max_rK, n_grid)

    PSIS, MS, RKS = np.meshgrid(Psis, Ms, rKs)

    PSIS = np.ndarray.flatten(PSIS)
    MS = np.ndarray.flatten(MS)
    RKS = np.ndarray.flatten(RKS)
    
    results = []
    
    ###Iterate through the grid and append the total log-likelihood to  a list
    for i in range(len(PSIS)):
        print(i)
        theta = np.array([MS[i], RKS[i], PSIS[i]])
        log_ls = log_likelihoods(theta, rs, vs)
        results.append(np.sum(log_ls))

    results = np.array(results)
    
    ###Each processor finds it's maximum likelihood value...
    max_log_l = np.max(results)
    max_idx = np.where(results == max_log_l)[0][0]
    
    ### ...and reports it back to the root process
    to_save = np.array([MS[max_idx], RKS[max_idx], PSIS[max_idx], max_log_l])
    recv_buff = None
    
    if (rank == 0):
        recv_buff = np.empty([size, 4], dtype = float)
        print(recv_buff)

    comm.Gather(to_save, recv_buff, root = 0)
    
    ### The root process compares the results form each processor, picks the largest value, and saves the output.
    if(rank == 0):
        print(recv_buff)
        max_logs = recv_buff[:,-1]
        max_idx = np.where(max_logs == np.max(max_logs))[0][0]
        
        array_to_save = recv_buff[max_idx, :]
        print(array_to_save)
        save_file = f'/home/s1984454/LoKi-fit/Data/max_likelihood_params_region_{region}.txt'
        #save_file = f'/home/s1984454/Desktop/LoKi-Fit/Data/max_likelihood_params_region_{region}.txt'
        with open(save_file, 'wb') as f:
            np.savetxt(f, array_to_save, delimiter= ' ')

    
    

    
    



































