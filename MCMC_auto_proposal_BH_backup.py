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
from LoKi_samp import LoKi_samp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import sys

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
    eps_min = prior_args[6]
    eps_max = prior_args[7]
    mu_min = prior_args[8]
    mu_max = prior_args[9]
    
    M0 = x[0]
    rK = x[1]
    Psi = x[2]
    eps = x[3]
    mu = x[4]
    
    a0 = Psi - 9*mu/(4*np.pi*eps)

    M0_flag = (M0 >= M0_min and M0 <= M0_max)
    rK_flag = (rK >= rK_min and rK <= rK_max)
    Psi_flag = (Psi >= Psi_min and Psi <= Psi_max)
    eps_flag = (eps >= eps_min and eps <= eps_max)
    mu_flag = (mu >= mu_min and mu <= mu_max)
    a0_flag = a0 > 0
    
    if (M0_flag and rK_flag and Psi_flag and a0_flag and mu_flag and eps_flag):
        return 0
    else:
        return -np.inf

def log_likelihood(theta, data):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    eps = theta[3]
    mu = theta[4]
    
    a0 = Psi - 9*mu/(4*np.pi*eps)
    
    if (a0<0):
        return -np.inf
    
    else:
        rs = data[:,0]
        vs = data[:,1]
        
        model = LoKi(mu, eps, Psi, pot_only = True)
        
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        
        ### Calculate dimensional scales.
        a = (9 * rK * Mhat)/(4*np.pi*G*M)
        Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat) 
        
        ### Caclulate individual star likelihoods        
        rhats = rs/rK
        vhats = vs*np.sqrt(a)
    
        # vhat_rhats = np.interp(rhats, model.rhat, np.sqrt(2*model.psi))
        
        # point_comparisons = vhats <=  vhat_rhats
        
        # if False in point_comparisons:
        #     return -np.inf
            
        # else:            
        psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
        Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
            
        ls = Ae/M * (np.exp(-Ehats) - 1)
        
        log_ls = np.log(ls)
        
        return np.sum(log_ls)
    
def metropolis_sampling(data, initial_parameters, covariance, nsamp, prior_args, rank, comm, save_samples = False):

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    size = comm.Get_size()
        
    current_parameters = initial_parameters
    proposal_parameters = np.empty(5)
    
    
    if (rank == 0):    
        
        samples = []                                         
        
        data = np.array_split(data,size)
        
        l_recv = np.empty(5)
        l_send = np.empty(5)
        accepted = 0
        rejected = 0
    
    else:   
        data = None
        l_recv = np.empty(5)
        l_send = np.empty(5)
        
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
            
            proposal_parameters = current_parameters + np.random.multivariate_normal(mean = [0,0,0,0,0], cov = covariance)            
            
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
            if(i%100 == 0):
                print(f'Iteration {i} done.')
                sys.stdout.flush()
            
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
        print(f'Accept rate = {acceptance_rate}')
        sys.stdout.flush()
        
        if(save_samples):
            
            np.savetxt(f'Data/SAMPLES_MCMC_auto_proposal_M_{M_true}_rK_{rK_true}_Psi_{Psi_true}_eps_{eps_true}_mu_{mu_true}.txt', samples)
            
        return acceptance_rate
    
    else: 
        
        return 0

def tune_covariance(data, initial_parameters, cov0, nsamp_tune, prior_args, target_acceptance_rate, acceptance_rate_tol, rank1, comm1):
    
    # comm1 = MPI.COMM_WORLD
    # rank1 = comm1.Get_rank()
    #print(f'I am processor {rank1}')
    sys.stdout.flush()
    cov_tuned = False
    
    while(cov_tuned == False):
        
        acceptance_rate = metropolis_sampling(data, initial_parameters, cov0, nsamp_tune, prior_args, rank1, comm1, save_samples = False)
        
        if(rank1 == 0):
        
            if(abs(acceptance_rate  - target_acceptance_rate) > acceptance_rate_tol):
                cov0 = cov0 * acceptance_rate / target_acceptance_rate
            else:
                cov_tuned = True
            #print(acceptance_rate)
            #sys.stdout.flush()
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
        sys.stdout.flush()
                        
    return models

def generate_data(n_stars, M_true, rK_true, Psi_true, eps_true, mu_true):
    
    sampling = dimensional_data_generation(n_stars, M_true, rK_true, Psi_true, mu_true, eps_true, save = False, validate = False)
    rs = np.sqrt(sampling.x**2 + sampling.y**2 + sampling.z**2)
    vs = np.sqrt(sampling.vx**2 + sampling.vy**2 + sampling.vz**2)

    data = np.zeros((n_stars,2))
    data[:,0] = rs
    data[:,1] = vs  
    
    return data

def initialise_proposal_covariance(M_true, rK_true, Psi_true, eps_true, mu_true):
    
    covariance = 0.001*np.identity(5)
    covariance[0,0] *= 100#M_true # M0
    covariance[1,1] *= 1#rK_true  # rK
    covariance[2,2] *= 1#Psi_true  # Psi
    covariance[3,3] *= 1#eps_true 
    covariance[4,4] *= 1#Psi_true 
    
    return covariance

G = 4.3009e-3

M_true = 500
rK_true = 1.2
Psi_true = 5
mu_true = 0.3
eps_true = 0.1

n_stars = 10000
nsamp = 100000
nsamp_tune = 1000

target_acceptance_rate = 0.236
acceptance_rate_tol = 0.02

M_max = 700
M_min = 300
rK_max = 2
rK_min = 0.5
Psi_max = 9
Psi_min = 1
eps_min = 0.01
eps_max = 1
mu_min = 0
mu_max = 0.5

prior_args = np.array([M_max, M_min, rK_max, rK_min, Psi_max, Psi_min, eps_min, eps_max, mu_min, mu_max])
initial_parameters =  np.array([M_true, rK_true, Psi_true, eps_true, mu_true]) + np.array([0,0,0,0,0])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if(rank == 0):
    print('Generating data...')
    sys.stdout.flush()
    data = generate_data(n_stars, M_true, rK_true, Psi_true, eps_true, mu_true)
    print('...done.')
    sys.stdout.flush()
else:
    data = None
comm.Barrier()


if(rank == 0):
    print('Initialising covariance...')
    sys.stdout.flush()
cov0 = initialise_proposal_covariance(M_true, rK_true, Psi_true, eps_true, mu_true)

if(rank == 0):
    print('...done.')
    sys.stdout.flush()
    
comm.Barrier()

if(rank == 0):
    print('Tuning covariance...')
    sys.stdout.flush()
    
covariance = tune_covariance(data, initial_parameters, cov0, nsamp_tune, prior_args, target_acceptance_rate, acceptance_rate_tol, rank, comm)

comm.Barrier()

if(rank == 0 ):
    print('..done.')
    sys.stdout.flush()
    print('Running MH algorithm...')
    sys.stdout.flush()
    
acceptance_rate = metropolis_sampling(data, initial_parameters, covariance, nsamp, prior_args, save_samples = True)

if(rank == 0):
    print('done.')
    sys.stdout.flush()





# model =  LoKi(mu_true, eps_true ,Psi_true)

# dimensionless_samples = []

# while (len(dimensionless_samples) < n_stars):
    
#     current_sample = LoKi_samp(model, N = 1, plot = False, scale_nbody = False)
    
#     xhat = current_sample.x[0]
#     yhat = current_sample.y[0]
#     zhat = current_sample.z[0]
#     vxhat = current_sample.vx[0]
#     vyhat = current_sample.vy[0]
#     vzhat = current_sample.vz[0]

#     v_t = np.sqrt(vxhat**2 + vyhat**2)
#     vhat = np.sqrt(vxhat**2 + vyhat**2 + vzhat**2)
#     rhat = np.sqrt(xhat**2 + yhat**2 + zhat**2)
#     psi_interp = interp1d(model.rhat, model.psi, fill_value = (Psi_true, 0), bounds_error = False)

#     Jhat = v_t * rhat
#     Ehat = np.clip(0.5* vhat**2 - psi_interp(rhat), a_max = 0, a_min = None)

#     def f(r, Ehat, Jhat, psi_interp):
        
#         function = 2*(Ehat + psi_interp(r)) - (Jhat/r)**2
        
#         return function
    
#     def f1(r, Ehat, Jhat, Psi_true):
        
#         function = 2*(Ehat + Psi_true) - (Jhat/r)**2
        
#         return function

#     try:
#         r_grid = np.linspace(model.rhat[0], model.rhat[-1], 10000)
#         function = 2*(Ehat + psi_interp(r_grid)) - (Jhat/r_grid)**2
        
        
#         # fig,ax = plt.subplots(1,1)
#         # ax.plot(r_grid, function)
#         # ax.set_ylim(-0.1,1)

#         upper_bracket = r_grid[np.where(function == max(function))[0][0]]
#         r_p = root_scalar(f = f, args = (Ehat, Jhat, psi_interp), bracket = [model.rhat[0], upper_bracket], method = 'bisect').root
        
#     except:
#         print('Tried to sample a point without a resolved r_p, retrying calculation with finer radial grid.')
#         r_grid = np.linspace(model.rhat[0], model.rhat[-1], 1000000)
#         function = 2*(Ehat + psi_interp(r_grid)) - (Jhat/r_grid)**2
#         upper_bracket = r_grid[np.where(function == max(function))[0][0]]
#         r_p = root_scalar(f = f, args = (Ehat, Jhat, psi_interp), bracket = [model.rhat[0], upper_bracket], method = 'bisect').root
        
        
#     if (r_p >= eps_true):
#         dimensionless_samples.append([xhat, yhat, zhat, vxhat, vyhat, vzhat])
#         #print(len(self.dimensionless_samples))

# dimensionless_samples = np.array(dimensionless_samples)
