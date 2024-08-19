#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:07:32 2023

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import gammainc, gamma
from mpi4py import MPI

def rho_hat(psi):
    
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
    density = np.nan_to_num(density,copy = False)
        
    return density
    
def log_prior(x, prior_args):
    
    G = 4.3009e-3
    
    M_max = prior_args[0]
    M_min = prior_args[1]
    rK_max = prior_args[2]
    rK_min = prior_args[3]
    Psi_max = prior_args[4]
    Psi_min = prior_args[5]
    
    M = x[0]
    rK = x[1]
    Psi = x[2]
       
    M_flag = (M >= M_min and M <= M_max)
    rK_flag = (rK >= rK_min and rK <= rK_max)
    Psi_flag = (Psi >= Psi_min and Psi <= Psi_max)
    
    # print(f'rho flag = {rho0_flag}')
    # print(f'rK flag = {rK_flag}')
    # print(f'Psi flag = {Psi_flag}')
    # print(f'a0 flag = {a0_flag}')
    
    if (M_flag and rK_flag and Psi_flag):
        V = 1
        return -np.log(V)
    else:
        return -np.inf
    
def generate_observed_from_true(data_path, fname):
    data = np.loadtxt(data_path + fname)
    
    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]
    vxs = data[:,3]
    vys = data[:,4]
    vzs = data[:,5]
        
    for i in range(int(len(data)/3)):
        
        zs[3*i] = None
        zs[3*i + 1] = None
        vxs[3*i + 1] = None
        vys[3*i + 1] = None
    
    observed_data = np.zeros((len(data),6))
    observed_data[:,0] = xs
    observed_data[:,1] = ys
    observed_data[:,2] = zs
    observed_data[:,3] = vxs
    observed_data[:,4] = vys
    observed_data[:,5] = vzs
    
    return observed_data

def split_data(observed_data):
    
    data_6d = []
    data_5d = []
    data_3d = []
    
    for i in range(len(observed_data)):
        row = observed_data[i,:]
        
        if np.isnan(row[3]):
            data_3d.append(row)
        
        elif(np.isnan(row[2])):
            data_5d.append(row)
            
        else:
            data_6d.append(row)
    
    data_3d = np.array(data_3d)
    data_5d = np.array(data_5d)
    data_6d = np.array(data_6d)
    
    return data_3d, data_5d, data_6d

def log_likelihood_6d(data_6d, parameters, args):  
    
    M = parameters[0]
    rK = parameters[1]
    Psi = parameters[2]
    G = 4.3009e-3
    
    model, Ae, a = args
    
    xs = data_6d[:,0].copy()
    ys = data_6d[:,1].copy()
    zs = data_6d[:,2].copy()
    vxs = data_6d[:,3].copy()
    vys = data_6d[:,4].copy()
    vzs = data_6d[:,5].copy()
      
    xs *= 1/rK
    ys *= 1/rK
    zs *= 1/rK
    vxs *= np.sqrt(a)
    vys *= np.sqrt(a)
    vzs *= np.sqrt(a)
        
    rhats = np.sqrt(xs**2 + ys**2 + zs**2)
    vhats = np.sqrt(vxs**2 + vys**2 + vzs**2)
    
    psi = np.interp(rhats, xp = model.rhat, fp = model.psi)
    Ehats = np.clip(0.5* vhats**2 - psi, a_max = 0, a_min = None)
    
    likelihoods = (Ae)/M * (np.exp(-Ehats) - 1)
    
    
    if (0 in likelihoods):
        return -np.inf
    
    else:
        return np.sum(np.log(likelihoods)) 

def log_likelihood_5d(data_5d, parameters, args, npoints):   
    
    M = parameters[0]
    rK = parameters[1]
    Psi = parameters[2]
    
    model, Ae, a = args
    G = 4.3009e-3
    
    xs = data_5d[:,0].copy()
    ys = data_5d[:,1].copy()
    vxs = data_5d[:,3].copy()
    vys = data_5d[:,4].copy()
    vzs = data_5d[:,5].copy()
    
    xs *= 1/rK
    ys *= 1/rK
    vxs *= np.sqrt(a)
    vys *= np.sqrt(a)
    vzs *= np.sqrt(a)
    
    x_perps = np.sqrt(xs**2 + ys**2)
    vs = np.sqrt(vxs**2 + vys**2 + vzs**2)
    
    rt = model.rhat[-1]
    zt_squared = rt**2 - x_perps**2
    
    if(np.any(zt_squared < 0)):
        return -np.inf
    
    zts = np.sqrt(zt_squared)
    
    z_grids =  np.linspace(0,zts,npoints)
    
    x_perp_matrix = np.tile(np.reshape(x_perps,(1,len(x_perps))),(npoints,1))
    v_matrix = np.tile(np.reshape(vs,(1,len(vs))),(npoints,1))
    
    r_grids =  np.sqrt(z_grids**2 + x_perp_matrix**2)
    
    psi_matrix = np.interp(x = r_grids, xp = model.rhat, fp = model.psi)
    
    E_hats = np.clip(0.5 * v_matrix**2 - psi_matrix, a_max = 0, a_min = None)
    integrand = np.exp(-E_hats) - 1
    
    likelihoods = (2*rK*Ae/M) * simps(y = integrand, x = z_grids, axis = 0)
    
    if (0 in likelihoods):
        return -np.inf
    
    else:
        return np.sum(np.log(likelihoods))
    
def log_likelihood_3d(data_3d, parameters, args, npoints):
    
    M = parameters[0]
    rK = parameters[1]
    Psi = parameters[2]
    
    model, Ae, a = args
    G = 4.3009e-3
    
    xs = data_3d[:,0].copy()
    ys = data_3d[:,1].copy()
    vzs = data_3d[:,5].copy()
    
    xs *= 1/rK
    ys *= 1/rK
    vzs *= np.sqrt(a)
    
    x_perps = np.sqrt(xs**2 + ys**2)
    x_perp_matrix = np.tile(np.reshape(x_perps,(1,len(x_perps))),(npoints,1))
    
    rt = model.rhat[-1]
    zt_squared = rt**2 - x_perps**2
    
    if(np.any(zt_squared < 0)):
        return -np.inf
    
    zts = np.sqrt(zt_squared)
    z_grids =  np.linspace(0,zts,npoints)
    
    r_grids =  np.sqrt(z_grids**2 + x_perp_matrix**2)
    psi_matrix = np.interp(x = r_grids, xp = model.rhat, fp = model.psi)
    
    vz_matrix = np.tile(np.reshape(vzs,(1,len(vzs))),(npoints,1))
    
    E = np.clip(psi_matrix - 0.5*vz_matrix**2, a_max = None, a_min = 0)
    integrand = np.exp(E)*gamma(2)*gammainc(2,E)
    
    likelihoods = (4*np.pi*Ae*rK/(M*a)) * simps(y = integrand, x = z_grids, axis = 0)
    
    if (0 in likelihoods):
        return -np.inf
    
    else:
        return np.sum(np.log(likelihoods)) 
    
def important_constants(parameters):
    
    M = parameters[0]
    rK = parameters[1]
    Psi = parameters[2]
    
    model = LoKi(0, 1e-6, Psi, pot_only = True)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
    
    return [model, Ae, a]

def metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp, prior_args, npoints, idx, save_samples = False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    current_parameters = initial_parameters
    proposal_parameters = np.empty(3)
    
    
    if (rank == 0):    
        
        samples = []                                         
    
        data = generate_observed_from_true(data_path, fname)        
        data = np.array_split(data,size)
        
        l_recv = np.empty(3)
        l_send = np.empty(3)
        accepted = 0
        rejected = 0
    
    else:   
        data = None
        l_recv = np.empty(3)
        l_send = np.empty(3)
        
    data = comm.scatter(data,root = 0)
    
    data_3d, data_5d, data_6d = split_data(data)
    #print(len(data_6d))
    
    
    l0_prior = log_prior(current_parameters,prior_args)
    
    assert l0_prior != -np.inf
    
    print(current_parameters)
    args = important_constants(current_parameters) 
    
    l_6d = log_likelihood_6d(data_6d, current_parameters, args)
    l_5d = log_likelihood_5d(data_5d, current_parameters, args, npoints)
    l_3d = log_likelihood_3d(data_3d, current_parameters, args, npoints)
    
    l_send = np.array([l_3d, l_5d, l_6d])
    
    comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
    
    if (rank == 0):
        
        l0 = np.sum(l_recv) + l0_prior
        #print(f'first likelihood = {l0}')
        
    for i in range(nsamp):
        
        if(rank == 0):
            
            proposal_parameters = current_parameters + np.random.multivariate_normal(mean = [0,0,0], cov = covariance)            
            
        proposal_parameters =  comm.bcast(proposal_parameters, root = 0)
        
        l_prior =  log_prior(proposal_parameters, prior_args)
        
        #if(rank==0):
            #print(f'proposal l_prior = {l_prior}')
        
        if(l_prior == -np.inf):
            
            l_send = np.array([-np.inf,-np.inf,-np.inf])
            
        else:
            
            args = important_constants(proposal_parameters) 
            
            l_6d = log_likelihood_6d(data_6d, proposal_parameters, args)
            l_5d = log_likelihood_5d(data_5d, proposal_parameters, args, npoints)
            l_3d = log_likelihood_3d(data_3d, proposal_parameters, args, npoints)
            
            l_send = np.array([l_3d, l_5d, l_6d])
        
        comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
    
        if(rank == 0):
            l = np.sum(l_recv) + l_prior
            #print(f'proposal l = {l}')
            if(i%1000 == 0):
                print(i)
            
            if (l == -np.inf):
                acceptance_prob = 0
            else:
                log_diff = l - l0
                acceptance_prob = min(1,np.exp(log_diff))
                
            if(np.random.rand() < acceptance_prob):
                
                samples.append(proposal_parameters)
                current_parameters = proposal_parameters
                l0 = l
                accepted += 1
                #print('accepted')
                
            else:
                samples.append(current_parameters)
                rejected +=1
                #print('rejected')
            
                
        comm.Barrier()
    
    if(rank == 0):
        
        acceptance_rate = accepted/(accepted+rejected)
        print(acceptance_rate)
        
        if(save_samples):
            
            np.savetxt(data_path + f'SAMPLES_incomplete_data_{idx}.txt', samples)
            
        return acceptance_rate
    
    else: 
        
        return 0

def tune_covariance(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, npoints, target_acceptance_rate, acceptance_rate_tol, idx):
    
    comm1 = MPI.COMM_WORLD
    rank1 = comm1.Get_rank()
    #print(rank1)
    cov_tuned = False
    
    while(cov_tuned == False):
        
        acceptance_rate = metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, npoints, idx, save_samples = False) 
        
        if(rank1 == 0):
        
            if(abs(acceptance_rate  - target_acceptance_rate) > acceptance_rate_tol):
                covariance = covariance * acceptance_rate / target_acceptance_rate
            else:
                cov_tuned = True
            print(acceptance_rate)
        cov_tuned = comm1.bcast(cov_tuned, root = 0)
        covariance = comm1.bcast(covariance, root = 0)

    return covariance

G = 4.3009e-3

M0 = 500
rK0 = 1.2
Psi0 = 5
mu0 = 0
eps0 = 1e-6

M_max = 700
M_min = 300
rK_max = 2
rK_min = 0.5
Psi_max = 9
Psi_min = 1

prior_args = np.array([M_max, M_min, rK_max, rK_min, Psi_max, Psi_min])

initial_parameters =  np.array([M0, rK0, Psi0]) + np.array([0, 0, 0])

print(log_prior(initial_parameters, prior_args))
 
data_path = '/home/s1984454/Desktop/LoKi-fit/Data/'
#data_path = '/home/s1984454/LoKi-fit/Data/'
fname = f'dimensional_samples_King_M_{M0}_rK_{rK0}_Psi_{Psi0}_mu_{mu0}_epsilon_{eps0}_N_20000.txt'
covariance = 0.01*np.identity(3)
covariance[0,0] *= 10 # M
covariance[1,1] *= 1  # rK
covariance[2,2] *= 1  # Psi

nsamp = 100000
nsamp_tune = 10000
npoints = 100

target_acceptance_rate = 0.236
acceptance_rate_tol = 0.02

covariance = tune_covariance(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, npoints, target_acceptance_rate, acceptance_rate_tol, 0)

for idx in range(1):
    print(f'run {idx} start.')
    acceptance_rate = metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp, prior_args, npoints, idx, save_samples = True)
    print(f'run {idx} end.')







































