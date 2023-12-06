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
    
def uniform_prior(x,lower_bound,upper_bound):
    
    if (x>=lower_bound and x<=upper_bound):
        
        return 1/(upper_bound - lower_bound)
    
    else:
        
        return 0

def log_prior(x, prior_args):
    
    G = 4.3009e-3
    
    Ae_max = prior_args[0]
    Ae_min = prior_args[1]
    rK_max = prior_args[2]
    rK_min = prior_args[3]
    Psi_max = prior_args[4]
    Psi_min = prior_args[5]
    
    Ae = x[0]
    rK = x[1]
    M_BH = x[2]
    Psi = x[3]
    epsilon = x[4]
    
    a = (32 * np.sqrt(2) * np.pi**2 * G * rho_hat(Psi) * rK**2 * Ae / 27)**2

    A_hat = 8 * np.sqrt(2) * np.pi * Ae / (3 * a**(3/2))

    mu = M_BH /(A_hat * rho_hat(Psi) * rK**3)
        
    a0 = Psi - (9*mu)/(4*np.pi*epsilon)
    
    Ae_flag = (Ae >= Ae_min and Ae <= Ae_max)
    rK_flag = (rK >= rK_min and rK <= rK_max)
    Psi_flag = (Psi >= Psi_min and Psi <= Psi_max)
    a0_flag = a0 >= 0
    M_BH_positive_flag = M_BH >= 0
    
    if (Ae_flag and rK_flag and Psi_flag and a0_flag and M_BH_positive_flag):
        V = 1
        return -np.log(V)
    else:
        return -np.inf
    
def generate_observed_from_true(data_path, fname):

    data = np.loadtxt(data_path + fname + '.txt')
    
    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]
    vxs = data[:,3]
    vys = data[:,4]
    vzs = data[:,5]

    
    observed_data = np.zeros((len(data),6))
    observed_data[:,0] = xs
    observed_data[:,1] = ys
    observed_data[:,2] = zs
    observed_data[:,3] = vxs
    observed_data[:,4] = vys
    observed_data[:,5] = vzs
    
    return data

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

def log_likelihood_6d(data_6d, parameters):  
    
    Ae = parameters[0]
    rK = parameters[1]
    M_BH = parameters[2]
    Psi = parameters[3]
    epsilon = parameters[4]
    G = 4.3009e-3
    
    a = (32 * np.sqrt(2) * np.pi**2 * G * rho_hat(Psi) * rK**2 * Ae / 27)**2
    A_hat = 8 * np.sqrt(2) * np.pi * Ae / (3 * a**(3/2))
    mu = M_BH /(A_hat * rho_hat(Psi) * rK**3)
    
    model = model = LoKi(mu, epsilon, Psi, pot_only = True)
    
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    
    M_scale = A_hat * rK**3 * rho_hat(Psi)
    
    M = Mhat * M_scale
    
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

def metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp, prior_args, save_samples = False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    current_parameters = initial_parameters
    proposal_parameters = np.empty(5)
    
    
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
    print(len(data_6d))
    
    
    l0_prior = log_prior(current_parameters,prior_args)
    
    assert l0_prior != -np.inf
    
    l_6d = log_likelihood_6d(data_6d, current_parameters)
    l_5d = 0
    l_3d = 0
    
    l_send = np.array([l_3d, l_5d, l_6d])
    
    comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
    
    if (rank == 0):
        
        l0 = np.sum(l_recv) + l0_prior
        
    for i in range(nsamp):
        
        if(rank == 0):
            
            proposal_parameters = current_parameters + np.random.multivariate_normal(mean = [0,0,0,0,0], cov = covariance)            
            
        proposal_parameters =  comm.bcast(proposal_parameters, root = 0)
        
        l_prior =  log_prior(proposal_parameters, prior_args)
        
        if(l_prior == -np.inf):
            
            l_send = np.array([-np.inf,-np.inf,-np.inf])
            
        else:
            
            l_6d = log_likelihood_6d(data_6d, proposal_parameters)
            l_5d = 0
            l_3d = 0
            
            l_send = np.array([l_3d, l_5d, l_6d])
        
        comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
    
        if(rank == 0):
            l = np.sum(l_recv) + l_prior
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
                
            else:
                samples.append(current_parameters)
                rejected +=1
            
                
        comm.Barrier()
    
    if(rank == 0):
        
        acceptance_rate = accepted/(accepted+rejected)
        print(acceptance_rate)
        
        if(save_samples):
            
            np.savetxt(data_path + 'SAMPLES_parallel_'+fname+'_fixed_eps_physical_mu.txt', samples)
            
        return acceptance_rate
    
    else: 
        
        return 0

def tune_covariance(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, target_acceptance_rate, acceptance_rate_tol):
    
    comm1 = MPI.COMM_WORLD
    rank1 = comm1.Get_rank()
    print(rank1)
    cov_tuned = False
    
    while(cov_tuned == False):
        
        acceptance_rate = metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, save_samples = False)
        
        if(rank1 == 0):
        
            if(abs(acceptance_rate  - target_acceptance_rate) > acceptance_rate_tol):
                covariance = covariance * acceptance_rate / target_acceptance_rate
            else:
                cov_tuned = True
            print(acceptance_rate)
        cov_tuned = comm1.bcast(cov_tuned, root = 0)
        covariance = comm1.bcast(covariance, root = 0)

    return covariance

def calculate_true_quantities(M, rK, Psi, mu, epsilon):
    
    G = 4.3009e-3
    
    true_model = LoKi(mu, epsilon, Psi, pot_only = True)
    Mhat = np.trapz(y = 4*np.pi*true_model.rhat**2 * true_model.density(true_model.psi) / true_model.density(true_model.Psi) , x = true_model.rhat)
    
    A_hat = M/(Mhat * rho_hat(Psi)*rK**3)

    a = 9/(4 * np.pi * G * A_hat * rho_hat(Psi) * rK**2)

    Ae = 3 * a**(3/2) * A_hat / (8 * np.sqrt(2) * np.pi)

    M_BH = mu * A_hat * rho_hat(Psi) * rK**3

    return Ae, M_BH

G = 4.3009e-3

M0 = 500
rK0 = 1.2
Psi0 = 5
mu0 = 0.3
eps0 = 0.1

Ae_max = 1
Ae_min = 0.01
rK_max = 2
rK_min = 0.5
Psi_max = 9
Psi_min = 1

Ae0, M_BH0 = calculate_true_quantities(M0, rK0, Psi0, mu0, eps0)

prior_args = np.array([Ae_max, Ae_min, rK_max, rK_min, Psi_max, Psi_min])

initial_parameters =  np.array([Ae0, rK0, M_BH0, Psi0, eps0])
 
#data_path = '/home/s1984454/Desktop/King_fitting/Data/'
data_path = '/home/s1984454/LoKi-fit/Data/'
fname = f'dimensional_samples_King_M_{M0}_rK_{rK0}_Psi_{Psi0}_mu_{mu0}_epsilon_{eps0}_N_20000' 
covariance = 0.01*np.identity(5)
covariance[0,0] *= 0.1
covariance[1,1] *= 1
covariance[2,2] *= 1
covariance[3,3] *= 1
covariance[4,4] = 0 # Remain at constant epsilon
nsamp = 100000
nsamp_tune = 10000

target_acceptance_rate = 0.2
acceptance_rate_tol = 0.02

covariance = tune_covariance(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, target_acceptance_rate, acceptance_rate_tol)

acceptance_rate = metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp, prior_args, save_samples = True)








































