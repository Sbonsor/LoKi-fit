#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:19:18 2024

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
    
def log_likelihood(theta, rs, vs, r1, r2):
    
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
        
        ### Calculate contribution due to mass enclosed in region.
        Mass_profile = model.M_r *M_scale
        r = model.rhat * rK
        mass_diff = np.interp(r2, xp = r, fp = Mass_profile) - np.interp(r1, xp = r, fp = Mass_profile)
        
        
        return -(N/M)*mass_diff + np.sum(log_ls)
    
def negative_log_likelihood(theta, rs, vs, r1, r2):
    return -log_likelihood(theta, rs, vs, r1, r2)

def single_derivative_log_l(i, h, rs, vs, r1, r2, theta):
    
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
 
    u_x2 = log_likelihood(theta + 2*x_increment, rs, vs, r1, r2)
    u_x1 = log_likelihood(theta + 1*x_increment, rs, vs, r1, r2)
    u_1x = log_likelihood(theta - 1*x_increment, rs, vs, r1, r2)
    u_2x = log_likelihood(theta - 2*x_increment, rs, vs, r1, r2)
    
    first_deriv = (-(u_x2/12) + (2*u_x1/3) - (2*u_1x/3) + (u_2x/12))/h
    
    return first_deriv

def info_matrix_single(region_data, region_boundaries, region):
    
    r1 = region_boundaries[region]
    r2 = region_boundaries[region+1]
    rs, vs = r_v(region_data)
    
    ##find theta_star for this set of data and region.
    minimisation = minimize(negative_log_likelihood, x0 = theta0, args = (rs, vs, r1, r2), method = 'Nelder-Mead', bounds = parameter_bounds)
    theta_star = minimisation.x

    ### calculate derivatives
    single_derivatives = np.zeros(3)
    for i in range(3):
        single_derivatives[i] = single_derivative_log_l(i, h, rs, vs, r1, r2, theta_star)
    ### Calculate I_ij    
    information_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            information_matrix[i][j] = information_matrix[j][i] = single_derivatives[i]*single_derivatives[j]
    
    return information_matrix

def extract_random_subsample(full_sample, N_sample):
    
    sub_sample_indices = np.random.choice(len(full_sample), N_sample, replace = False)
    sub_sample = full_sample[sub_sample_indices,:]

    return sub_sample

### Set parameters

M = 500
rK = 1.2
Psi = 5
n = 10000
M_min = 450
M_max = 550
rK_min = 1 
rK_max = 1.4
Psi_min = 3
Psi_max = 7
h = 1e-4
region = 2
n_points = 100
data_path = '/home/s1984454/LoKi-fit/Data/dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000.txt'
#data_path = '/home/s1984454/Desktop/LoKi-Fit/Data/dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000.txt'

### Derived parameters
data = np.loadtxt(data_path)
N = len(data)
parameter_bounds = [(M_min, M_max), (rK_min, rK_max), (Psi_min, Psi_max)]
theta0 = np.array([M, rK, Psi]) 
region_boundaries = radii_separations(Psi, rK)
save_path = f'/home/s1984454/LoKi-fit/Data/M_{M}_rK_{rK}_Psi_{Psi}_N_{N}_n_{n}_region_{region}_information_matrix.txt'
#save_path = f'/home/s1984454/Desktop/LoKi-Fit/Data/M_{M}_rK_{rK}_Psi_{Psi}_N_{N}_n_{n}_region_{region}_information_matrix.txt'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

split_data = split_data_by_radius(data, region_boundaries)  
region_data = split_data[region]

### Matrix to accumulate result within a single processor, and one to act as the receive buffer for the total matrix.
info_mat_combined = np.zeros((3,3))
total_matrix = np.zeros((3,3))

### Each processor calculates n_points matrices from sub samples of size n
for idx in range(n_points):
    
    sub_sample_region_data = extract_random_subsample(region_data, n)
    info_mat_single = info_matrix_single(region_data, region_boundaries, region)

    info_mat_combined = info_mat_combined + info_mat_single
    
    if(rank == 0):
        print(idx)

###Sum all calculated matrices.
comm.Reduce(info_mat_combined, total_matrix, root = 0, op = MPI.SUM )

### Divide by the total number of calculated matrices, and save the result.
if(rank == 0):

    total_matrix = total_matrix/ (n_points * size)
    
    np.savetxt(save_path, total_matrix)







      
        
        
        
        
        
        
        
        
        
        
        
        
