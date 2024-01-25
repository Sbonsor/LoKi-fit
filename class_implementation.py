#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:13:07 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import gammainc, gamma
from mpi4py import MPI

class fitting:
    
    def __init__(self, true_params, prior_args, initial_offset, data_path, data_file, nsamp, nsamp_tune, target_acceptance_rate, acceptance_rate_tol, save_samples, output_file,  **kwargs ):
        
        self._set_kwargs(true_params, prior_args, initial_offset, data_path, data_file, nsamp, nsamp_tune, target_acceptance_rate, acceptance_rate_tol, save_samples, output_file, **kwargs)
        self.tune_covariance()
        self.metropolis_sampling(self.nsamp, self.save_samples)
        
    def _set_kwargs(self, true_params, prior_args, initial_offset, data_path, data_file, nsamp, nsamp_tune, target_acceptance_rate, acceptance_rate_tol, save_samples, output_file,  **kwargs):
        
        self.G = 4.3009e-3
        
        self.initial_offset = initial_offset
        
        self.data_path = data_path
        self.data_file = data_file
        self.output_file = output_file
        self.nsamp = nsamp
        self.nsamp_tune = nsamp_tune
        self.target_acceptance_rate = target_acceptance_rate
        self.acceptance_rate_tol = acceptance_rate_tol
        self.save_samples = save_samples
        
        self.M0 = true_params[0]
        self.rK0 = true_params[1]
        self.Psi0 = true_params[2]
        self.mu0 = true_params[3]
        self.eps0 = true_params[4]

        self.rho0_max = prior_args[0]
        self.rho0_min = prior_args[1]
        self.rK_max = prior_args[2]
        self.rK_min = prior_args[3]
        self.Psi_max = prior_args[4]
        self.Psi_min = prior_args[5]

        self.rho00, self.M_BH0 = self.calculate_true_quantities()
        
        
        self.initial_parameters =  true_params + self.initial_offset

        print(self.log_prior(self.initial_parameters))
         
        self.covariance = 0.01*np.identity(5)
        self.covariance[0,0] *= 1 # rho_0
        self.covariance[1,1] *= 1  # rK
        self.covariance[2,2] *= 0  # M_BH
        self.covariance[3,3] *= 1  # Psi
        self.covariance[4,4] *= 0  # epsilon
                
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
        
        
    def rho_hat(self, psi):
        
        density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
        density = np.nan_to_num(density,copy = False)
            
        return density
        
    def log_prior(self, x):
        
        rho_0 = x[0]
        rK = x[1]
        M_BH = x[2]
        Psi = x[3]
        epsilon = x[4]
        
        mu = M_BH /(rho_0* rK**3)
        a0 = Psi - (9*mu)/(4*np.pi*epsilon)
        
        rho0_flag = (rho_0 >= self.rho0_min and rho_0 <= self.rho0_max)
        rK_flag = (rK >= self.rK_min and rK <= self.rK_max)
        Psi_flag = (Psi >= self.Psi_min and Psi <= self.Psi_max)
        a0_flag = a0 >= 0
        M_BH_positive_flag = M_BH >= 0
        
        if (rho0_flag and rK_flag and Psi_flag and a0_flag and M_BH_positive_flag):
            V = 1
            return -np.log(V)
        else:
            return -np.inf
        
    def generate_observed_from_true(self):

        data = np.loadtxt(self.data_path + self.data_file + '.txt')
        
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

    def split_data(self, observed_data):
        
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

    def log_likelihood_6d(self, data_6d, parameters):  
        
        rho_0 = parameters[0]
        rK = parameters[1]
        M_BH = parameters[2]
        Psi = parameters[3]
        epsilon = parameters[4]
        G = 4.3009e-3
        
        mu = M_BH /(rho_0* rK**3)
        a = 9/(4 * np.pi * G * rho_0 * rK**2)
        Ae = (3 * a**(3/2) * rho_0)/ (8 * np.sqrt(2) * np.pi * self.rho_hat(Psi))
        
        model = LoKi(mu, epsilon, Psi, pot_only = True)
        
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        
        M_scale = rho_0 * rK**3 
        
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

    def metropolis_sampling(self, nsamp, save_samples):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        current_parameters = self.initial_parameters
        proposal_parameters = np.empty(5)
        
        
        if (rank == 0):    
            
            samples = []                                         
        
            data = self.generate_observed_from_true()        
            data = np.array_split(data, size)
            
            l_recv = np.empty(3)
            l_send = np.empty(3)
            accepted = 0
            rejected = 0
        
        else:   
            data = None
            l_recv = np.empty(3)
            l_send = np.empty(3)
            
        data = comm.scatter(data,root = 0)
        
        data_3d, data_5d, data_6d = self.split_data(data)
        print(len(data_6d))
        
        
        l0_prior = self.log_prior(current_parameters)
        
        assert l0_prior != -np.inf
        
        l_6d = self.log_likelihood_6d(data_6d, current_parameters)
        l_5d = 0
        l_3d = 0
        
        l_send = np.array([l_3d, l_5d, l_6d])
        
        comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
        
        if (rank == 0):
            
            l0 = np.sum(l_recv) + l0_prior
            
        for i in range(nsamp):
            
            if(rank == 0):
                
                proposal_parameters = current_parameters + np.random.multivariate_normal(mean = [0,0,0,0,0], cov = self.covariance)            
                
            proposal_parameters =  comm.bcast(proposal_parameters, root = 0)
            
            l_prior =  self.log_prior(proposal_parameters)
            
            if(l_prior == -np.inf):
                
                l_send = np.array([-np.inf,-np.inf,-np.inf])
                
            else:
                
                l_6d = self.log_likelihood_6d(data_6d, proposal_parameters)
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
                
                np.savetxt(self.data_path + self.output_file, samples)
                
            return acceptance_rate
        
        else: 
            
            return 0

    # def tune_covariance(self):
        
    #     comm1 = MPI.COMM_WORLD
    #     rank1 = comm1.Get_rank()
    #     print(rank1)
    #     cov_tuned = False
        
    #     while(cov_tuned == False):
            
    #         acceptance_rate = metropolis_sampling(data_path, fname, initial_parameters, covariance, nsamp_tune, prior_args, save_samples = False)
            
    #         if(rank1 == 0):
            
    #             if(abs(acceptance_rate  - target_acceptance_rate) > acceptance_rate_tol):
    #                 covariance = covariance * acceptance_rate / target_acceptance_rate
    #             else:
    #                 cov_tuned = True
    #             print(acceptance_rate)
    #         cov_tuned = comm1.bcast(cov_tuned, root = 0)
    #         covariance = comm1.bcast(covariance, root = 0)

    #     return covariance
    
    def tune_covariance(self):
        
        cov_tuned = False
        
        while(cov_tuned == False):
            
            acceptance_rate = self.metropolis_sampling(self.nsamp_tune, save_samples = False)
        
            if(abs(acceptance_rate  - self.target_acceptance_rate) > self.acceptance_rate_tol):
                self.covariance = self.covariance * acceptance_rate / self.target_acceptance_rate
            else:
                cov_tuned = True
                
            print(acceptance_rate)
            
        return 1

    def calculate_true_quantities(self):
        
        true_model = LoKi(self.mu0, self.eps0, self.Psi0, pot_only = True)
        Mhat = np.trapz(y = 4*np.pi*true_model.rhat**2 * true_model.density(true_model.psi) / true_model.density(true_model.Psi) , x = true_model.rhat)
        
        rho_0 = self.M0 / (Mhat * self.rK0**3)
        
        M_BH = self.mu0 * rho_0 * self.rK0**3

        return rho_0, M_BH