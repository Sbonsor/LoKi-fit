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
np.random.seed(11294381)


def uniform_prior(x,lower_bound,upper_bound):
    
    if (x>=lower_bound and x<=upper_bound):
        
        return 1/(upper_bound - lower_bound)
    
    else:
        
        return 0


def log_prior(x, prior_args):
    
    M_max = prior_args[0]
    M_min = prior_args[1]
    rK_max = prior_args[2]
    rK_min = prior_args[3]
    Psi_max = prior_args[4]
    Psi_min = prior_args[5]
    epsilon_max = prior_args[6]
    epsilon_min = prior_args[7]
    
    M = x[0]
    rK = x[1]
    Psi = x[2]
    mu = x[3]
    epsilon = x[4]
    
    a0 = Psi - (9*mu)/(4*np.pi*epsilon)
    
    M_flag = (M >= M_min and M <= M_max)
    rK_flag = (rK >= rK_min and rK <= rK_max)
    Psi_flag = (Psi >= Psi_min and Psi <= Psi_max)
    epsilon_flag = (epsilon >= epsilon_min and epsilon <= epsilon_max)
    a0_flag = a0 >= 0
    mu_positive_flag = mu >= 0
    
    if (M_flag and rK_flag and Psi_flag and epsilon_flag and a0_flag and mu_positive_flag):
        V = np.pi * (M_max - M_min) * (rK_max - rK_min) * (Psi_max **2 - Psi_min **2) * (epsilon_max**2 - epsilon_min**2) / 9
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

def determine_model_Ae_a(parameters):
    
    M = parameters[0]
    rc = parameters[1]
    Psi = parameters[2]
    mu = parameters[3]
    epsilon = parameters[4]
    G = 4.3009e-3
    
    model = LoKi(mu, epsilon, Psi, pot_only = True)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    
    a = (9 * rc * Mhat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rc**3 * Mhat)
    
    return model, Ae, a

def log_likelihood_6d(data_6d,parameters,args):  
    
    M = parameters[0]
    rc = parameters[1]
    Psi = parameters[2]
    mu = parameters[3]
    epsilon = parameters[4]
    G = 4.3009e-3
    
    model, Ae, a = args
      
    xs = data_6d[:,0].copy()
    ys = data_6d[:,1].copy()
    zs = data_6d[:,2].copy()
    vxs = data_6d[:,3].copy()
    vys = data_6d[:,4].copy()
    vzs = data_6d[:,5].copy()
      
    xs *= 1/rc
    ys *= 1/rc
    zs *= 1/rc
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

def log_likelihood_5d(data_5d,parameters,args, npoints):   
    
    M = parameters[0]
    rc = parameters[1]
    Psi = parameters[2]
    mu = parameters[3]
    epsilon = parameters[4]
    
    model, Ae, a = args
    G = 4.3009e-3
    
    xs = data_5d[:,0].copy()
    ys = data_5d[:,1].copy()
    vxs = data_5d[:,3].copy()
    vys = data_5d[:,4].copy()
    vzs = data_5d[:,5].copy()
    
    xs *= 1/rc
    ys *= 1/rc
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
    
    likelihoods = (2*rc*Ae/M) * simps(y = integrand, x = z_grids, axis = 0)
    
    if (0 in likelihoods):
        return -np.inf
    
    else:
        return np.sum(np.log(likelihoods))
    
def log_likelihood_3d(data_3d, parameters, args, npoints):
    
    M = parameters[0]
    rc = parameters[1]
    Psi = parameters[2]
    mu = parameters[3]
    epsilon = parameters[4]
    
    model, Ae, a = args
    G = 4.3009e-3
    
    xs = data_3d[:,0].copy()
    ys = data_3d[:,1].copy()
    vzs = data_3d[:,5].copy()
    
    xs *= 1/rc
    ys *= 1/rc
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
    
    likelihoods = (4*np.pi*Ae*rc/(M*a)) * simps(y = integrand, x = z_grids, axis = 0)
    
    if (0 in likelihoods):
        return -np.inf
    
    else:
        return np.sum(np.log(likelihoods)) 

def metropolis_sampling(data_path, fname, M0_rc0_Psi0_mu0_eps0, covariance, nsamp, prior_args, npoints, save_samples = False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    current_parameters = M0_rc0_Psi0_mu0_eps0
    proposal_parameters = np.empty(5)
    
    
    if (rank == 0):    
        
        samples = []                                         
    
        data = generate_observed_from_true(data_path, fname)        
        data = np.array_split(data,size)
        
        args = determine_model_Ae_a(current_parameters)
        
        l_recv = np.empty(3)
        l_send = np.empty(3)
        accepted = 0
        rejected = 0
    
    else:   
        data = None
        args = None
        l_recv = np.empty(3)
        l_send = np.empty(3)
        
    args = comm.bcast(args, root = 0)
    data = comm.scatter(data,root = 0)
    
    data_3d, data_5d, data_6d = split_data(data)
    print(len(data_6d))
    
    
    l0_prior = log_prior(current_parameters,prior_args)
    
    assert l0_prior != -np.inf
    
    l_6d = log_likelihood_6d(data_6d, current_parameters, args)
    #l_5d = log_likelihood_5d(data_5d, current_parameters, args, npoints)
    #l_3d = log_likelihood_3d(data_3d, current_parameters, args, npoints)
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
            
            args = determine_model_Ae_a(proposal_parameters)
            
            l_6d = log_likelihood_6d(data_6d, proposal_parameters, args)
            #l_5d = log_likelihood_5d(data_5d, proposal_parameters, args, npoints)
            #l_3d = log_likelihood_3d(data_3d, proposal_parameters, args, npoints)
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
            
            np.savetxt(data_path + 'SAMPLES_parallel_'+fname+'_with_incomplete_data_and_mu.txt', samples)
            
        return acceptance_rate
    
    else: 
        
        return 0

def tune_covariance(data_path, fname, M0_rc0_Psi0, covariance, nsamp_tune, prior_args, npoints, target_acceptance_rate, acceptance_rate_tol):
    
    comm1 = MPI.COMM_WORLD
    rank1 = comm1.Get_rank()
    print(rank1)
    cov_tuned = False
    
    while(cov_tuned == False):
        
        acceptance_rate = metropolis_sampling(data_path, fname, M0_rc0_Psi0_mu0_eps0, covariance, nsamp_tune, prior_args, npoints, save_samples = False)
        
        if(rank1 == 0):
        
            if(abs(acceptance_rate  - target_acceptance_rate) > acceptance_rate_tol):
                covariance = covariance * acceptance_rate / target_acceptance_rate
            else:
                cov_tuned = True
            print(acceptance_rate)
        cov_tuned = comm1.bcast(cov_tuned, root = 0)
        covariance = comm1.bcast(covariance, root = 0)

    return covariance


M0 = 500
rc0 = 1.2
Psi0 = 5
mu0 = 0.3
eps0 = 0.1

M_max = 550
M_min = 450
rK_max = 2
rK_min = 0.5
Psi_max = 9
Psi_min = 1
epsilon_max = 0.3
epsilon_min = 1e-6

prior_args = np.array([M_max, M_min, rK_max, rK_min, Psi_max, Psi_min, epsilon_max, epsilon_min])
M0_rc0_Psi0_mu0_eps0 =  np.array([M0, rc0, Psi0, mu0, eps0])
#data_path = '/home/s1984454/Desktop/King_fitting/Data/'
data_path = '/home/s1984454/LoKi-fit/Data/'
fname = f'dimensional_samples_King_M_{M0}_rK_{rc0}_Psi_{Psi0}_mu_{mu0}_epsilon_{eps0}_N_20000' 
covariance = 0.01*np.identity(5)
covariance[0,0] *= 100
covariance[4,4] = 0 # Remain at constant epsilon
nsamp = 200000
npoints = 100
nsamp_tune = 10000

target_acceptance_rate = 0.2
acceptance_rate_tol = 0.02

covariance = tune_covariance(data_path, fname, M0_rc0_Psi0_mu0_eps0, covariance, nsamp_tune, prior_args, npoints, target_acceptance_rate, acceptance_rate_tol)

acceptance_rate = metropolis_sampling(data_path, fname, M0_rc0_Psi0_mu0_eps0, covariance, nsamp, prior_args, npoints, save_samples = True)








































