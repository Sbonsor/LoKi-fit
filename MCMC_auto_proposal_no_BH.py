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
from dimensional_data_generation import dimensional_data_generation

def rho_hat(psi):
    
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
    density = np.nan_to_num(density,copy = False)
        
    return density
    
def log_prior(x, prior_args):
    
    M0_max = prior_args[0]
    M0_min = prior_args[1]
    rK_max = prior_args[2]
    rK_min = prior_args[3]
    Psi_max = prior_args[4]
    Psi_min = prior_args[5]
    
    M0 = x[0]
    rK = x[1]
    Psi = x[2]


    M0_flag = (M0 >= M0_min and M0 <= M0_max)
    rK_flag = (rK >= rK_min and rK <= rK_max)
    Psi_flag = (Psi >= Psi_min and Psi <= Psi_max)

    
    if (M0_flag and rK_flag and Psi_flag):
        return 0
    else:
        return -np.inf

def log_likelihood(theta, data):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    
    rs = data[:,0]
    vs = data[:,1]
    
    model = LoKi(0, 1e-6, Psi, pot_only = True)
    
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    
    ### Calculate dimensional scales.
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat) 
    
    ### Caclulate individual star likelihoods        
    rhats = rs/rK
    vhats = vs*np.sqrt(a)

    vhat_rhats = np.interp(rhats, model.rhat, np.sqrt(2*model.psi))
    
    point_comparisons = vhats <=  vhat_rhats
    
    if False in point_comparisons:
        return -np.inf
        
    else:            
        psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
        Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
            
        ls = Ae/M * (np.exp(-Ehats) - 1)
        
        log_ls = np.log(ls)
    
        return np.sum(log_ls)
    
def metropolis_sampling(data, initial_parameters, covariance, nsamp, prior_args, save_samples = False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    current_parameters = initial_parameters
    proposal_parameters = np.empty(3)
    
    
    if (rank == 0):    
        
        samples = []                                         
        
        data = np.array_split(data,size)
        
        l_recv = np.empty(3)
        l_send = np.empty(3)
        accepted = 0
        rejected = 0
    
    else:   
        data = None
        l_recv = np.empty(3)
        l_send = np.empty(3)
        
    data = comm.scatter(data, root = 0)    
    l0_prior = log_prior(current_parameters, prior_args)
    
    assert l0_prior != -np.inf
    
    l_6d = log_likelihood(current_parameters, data)
    l_5d = 0
    l_3d = 0
    
    l_send = np.array([l_3d, l_5d, l_6d])
    
    comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
    
    if (rank == 0):
        
        l0 = np.sum(l_recv) + l0_prior
        
    for i in range(nsamp):
        
        if(rank == 0):
            
            proposal_parameters = current_parameters + np.random.multivariate_normal(mean = [0,0,0], cov = covariance)            
            
        proposal_parameters =  comm.bcast(proposal_parameters, root = 0)
        
        l_prior =  log_prior(proposal_parameters, prior_args)
        
        if(l_prior == -np.inf):
            
            l_send = np.array([-np.inf,-np.inf,-np.inf])
            
        else:
            
            l_6d = log_likelihood(proposal_parameters, data)
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
                np.seterr(over = 'ignore')
                acceptance_prob = min(1,np.exp(log_diff))
                np.seterr(over = 'warn')
                
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
            
            
            np.savetxt(f'Data/SAMPLES_MCMC_auto_proposal_M_{M_true}_rK_{rK_true}_Psi_{Psi_true}.txt', samples)
            
        return acceptance_rate
    
    else: 
        
        return 0

def tune_covariance(data, initial_parameters, cov0, nsamp_tune, prior_args, target_acceptance_rate, acceptance_rate_tol):
    
    comm1 = MPI.COMM_WORLD
    rank1 = comm1.Get_rank()
    print(rank1)
    cov_tuned = False
    
    while(cov_tuned == False):
        
        acceptance_rate = metropolis_sampling(data, initial_parameters, cov0, nsamp_tune, prior_args, save_samples = False)
        
        if(rank1 == 0):
        
            if(abs(acceptance_rate  - target_acceptance_rate) > acceptance_rate_tol):
                cov0 = cov0 * acceptance_rate / target_acceptance_rate
            else:
                cov_tuned = True
            print(acceptance_rate)
        cov_tuned = comm1.bcast(cov_tuned, root = 0)
        covariance = comm1.bcast(cov0, root = 0)

    return covariance

def precalculate_models(Psi_axis):
        
    models = {}
        
    for Psi in Psi_axis:
        model = LoKi(0, 1e-6, Psi, pot_only = False)
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        models[str(Psi)] = {'model' : model , 'Mhat' : Mhat}    
        print(Psi)
                        
    return models

def generate_data(n_stars, M_true, rK_true, Psi_true):
    
    sampling = dimensional_data_generation(n_stars, M_true, rK_true, Psi_true, 0, 1e-6, save = False, validate = False)
    rs = np.sqrt(sampling.x**2 + sampling.y**2 + sampling.z**2)
    vs = np.sqrt(sampling.vx**2 + sampling.vy**2 + sampling.vz**2)

    data = np.zeros((n_stars,2))
    data[:,0] = rs
    data[:,1] = vs  
    
    return data

def initialise_proposal_covariance(M_min, M_max, Psi_min, Psi_max, rK_min, rK_max, npoints_axis, data):
    
    rs = data[:,0]
    vs = data[:,1]
    
    M_axis = np.linspace(M_min, M_max, npoints_axis)
    Psi_axis = np.linspace(Psi_min, Psi_max, npoints_axis)
    rK_axis = np.linspace(rK_min, rK_max, npoints_axis)
    
    models = precalculate_models(Psi_axis)
    
    X,Y,Z = np.meshgrid(M_axis, rK_axis, Psi_axis)
    
    Ms = X.flatten()
    RKs = Y.flatten()
    PSIS = Z.flatten()
    
    log_ls = np.zeros(len(PSIS))
    
    for i in range(len(PSIS)):
        
        M = Ms[i]
        rK = RKs[i]
        Psi = PSIS[i]
        
        model = models[str(Psi)]['model']
        Mhat = models[str(Psi)]['Mhat']
            
        a = (9 * rK * Mhat)/(4*np.pi*G*M)
        Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat) 
            
        ### Caclulate individual star likelihoods        
        rhats = rs/rK
        vhats = vs*np.sqrt(a)

        vhat_rhats = np.interp(rhats, model.rhat, np.sqrt(2*model.psi))
    
        point_comparisons = vhats <=  vhat_rhats
        
        if False in point_comparisons:
            
            log_ls[i] = -np.inf
                     
        else:
            
            psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
            Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
                
            ls = Ae/M * (np.exp(-Ehats) - 1)
            
            log_ls[i] = np.sum(np.log(ls))
            
    
    mode_idx = np.where(log_ls == np.max(log_ls))[0][0]
    
    covariance = 0.01*np.identity(3)
    covariance[0,0] *= Ms[mode_idx]  # M0
    covariance[1,1] *= RKs[mode_idx]  # rK
    covariance[2,2] *= PSIS[mode_idx]  # Psi
    
    return covariance

G = 4.3009e-3

M_true = 10**6.05
rK_true = 0.51
Psi_true = 8.6

n_stars = 10000
nsamp = 100000
nsamp_tune = 10000
npoints_axis = 10

target_acceptance_rate = 0.236
acceptance_rate_tol = 0.02

M_max = 700
M_min = 300
rK_max = 2
rK_min = 0.5
Psi_max = 9
Psi_min = 1

prior_args = np.array([M_max, M_min, rK_max, rK_min, Psi_max, Psi_min])
initial_parameters =  np.array([M_true, rK_true, Psi_true]) + np.array([0,0,0])

print('Generating data...')
data = generate_data(n_stars, M_true, rK_true, Psi_true)
print('...done.')
print('Initialising covariance...')
cov0 = initialise_proposal_covariance(M_min, M_max, Psi_min, Psi_max, rK_min, rK_max, npoints_axis, data)
print('...done.')
print('Tuning covariance...')
covariance = tune_covariance(data, initial_parameters, cov0, nsamp_tune, prior_args, target_acceptance_rate, acceptance_rate_tol)
print('..done.')
print('Running MH algorithm...')
acceptance_rate = metropolis_sampling(data, initial_parameters, covariance, nsamp, prior_args, save_samples = True)
print('done.')