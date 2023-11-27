#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:08:06 2023

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import gammainc, gamma
from mpi4py import MPI
    
class metropolis_sampling:
    
    def __init__(self, **kwargs):
        print('got here')
        self._set_kwargs(**kwargs)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        if(self.rank == 0):
            print('Reading data...')
            
        self.read_data()
        
        if(self.rank == 0):
            print('done.')
            
        if(self.rank == 0):
            print('Tuning covariance...')
            
        self.tune_covariance()
        
        if(self.rank == 0):
            print('done.')
        
        if(self.rank == 0):
            print('Running chain...')
                
        self.run_chain()
        
        if(self.rank == 0):
            print('done.')
        
    def _set_kwargs(self, **kwargs):
        
        self.epsilon = 0.1
        self.G = 4.3009e-3
        self.Psi0_a00_M0_rK0 = [5,2.851408268259413, 500, 1.2]
        self.prior_args = [9, 1, 700, 300, 2, 0.5]
        self.data_path = '/home/s1984454/LoKi-fit/Data/'
        #self.data_path = '/home/s1984454/Desktop/LoKi-Fit/Data/'
        self.fname = f'dimensional_samples_King_M_{500}_rK_{1.2}_Psi_{5}_mu_{0.3}_epsilon_{0.1}_N_20000'
        self.nsamp = 10000
        self.nsamp_tune = 2000
        self.covariance = 0.01*np.identity(4)
        self.covariance[0,0] *= 100
        self.target_accept_prob = 0.2
        self.accept_prob_tol = 0.02
        
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self, key, value)
                
    def read_data(self):
        
        if (self.rank == 0):    
            
            data = np.loadtxt(self.data_path + self.fname + '.txt')        
            data = np.array_split(data, self.size)
     
        else: 
            
            data = None
        
        self.data = self.comm.scatter(data, root = 0)
           
        print(f'I am process {self.rank}, and I have data of length {len(self.data)}')
        
        return 1
    
    def l_prior(self, parameters):
        
        Psi, a0, M, rK = parameters
        Psi_max, Psi_min, M_max, M_min, rK_max, rK_min = self.prior_args
        
        V = 0.5 * (rK_max - rK_min) * (M_max - M_min) * (Psi_max**2 - Psi_min**2)
        
        log_prior = np.log(1/V)
        
        return log_prior
    
    def log_likelihood_6d(self, parameters):
        
        Psi, a0, M, rK = parameters
        
        mu = (Psi - a0) * 4*np.pi*self.epsilon/9
        
        ##calculate constant factors in DF definition.
        model = LoKi(mu, self.epsilon, Psi, pot_only = True)
        M_hat = -mu - 4*np.pi * model.rhat[-1]**2 * model.dpsi_dr[-1]/9
        a = (9 * rK * M_hat)/(4*np.pi*self.G*M)
        Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * M_hat)
        
        ### Take dimensional data and non-dimensionalise based on input parameter values/
        xs = self.data[:,0].copy()
        ys = self.data[:,1].copy()
        zs = self.data[:,2].copy()
        vxs = self.data[:,3].copy()
        vys = self.data[:,4].copy()
        vzs = self.data[:,5].copy()
          
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
        
        ### Return -infinity if there is a zero likelihood for a point. Otherwise take the logarithm and sum.
        if (0 in likelihoods):
            return -np.inf
        else:
            return np.sum(np.log(likelihoods))
        
    def run_chain(self, tuning = False):
        
        if(tuning):
            
            nsamp = self.nsamp_tune
            
        else:
            
            nsamp = self.nsamp
        
        current_parameters = self.Psi0_a00_M0_rK0
        proposal_parameters = np.empty(4)
        
        self.samples = []
        
        l_recv = np.empty([1], dtype = 'float')
        l_send = np.empty(1)
        
        self.accepted = 0
        self.rejected = 0
        
        l0_prior = self.l_prior(current_parameters)
        
        assert l0_prior != -np.inf
        
        l_6d = self.log_likelihood_6d(current_parameters)
        l_send = np.array([l_6d])
        self.comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
        
        if (self.rank == 0):
            
            l0 = l_recv[0] + l0_prior
        
        for i in range(nsamp):
            
            if (self.rank == 0):
                
                proposal_parameters = current_parameters + np.random.multivariate_normal(mean = [0,0,0,0], cov = self.covariance)
                
            proposal_parameters =  self.comm.bcast(proposal_parameters, root = 0)
            
            l_prior =  self.l_prior(proposal_parameters)
            
            if (l_prior == -np.inf):
                l_send = np.array([-np.inf])
                
            else:
                l_6d = self.log_likelihood_6d(proposal_parameters)
                l_send = np.array([l_6d])
            
            l_recv = np.empty([1], dtype = 'float')
            self.comm.Reduce(l_send, l_recv, MPI.SUM, root = 0)
            
            if (self.rank == 0):
                
                l = l_recv[0] + l_prior
                
                if(l == -np.inf):
                    
                    acceptance_probability = 0
                    
                else:
                    
                    log_diff =  l - l0
                    acceptance_probability = min(1, np.exp(log_diff))
                    
                    if (np.random.rand() < acceptance_probability):
                        
                        self.samples.append(proposal_parameters)
                        current_parameters = proposal_parameters
                        l0 = l
                        self.accepted += 1
                        
                    else:
                        
                        self.samples.append(current_parameters)
                        self.rejected += 1
                print(i)        
                self.comm.Barrier()
            self.comm.Barrier()
            
            if(self.rank == 0):
                
                self.acceptance_rate = self.accepted/(self.accepted + self.rejected)
                print(f'Acceptance rate is {self.acceptance_rate}')

                np.savetxt(f'{self.data_path}SAMPLES_parallel_{self.fname}_with_incomplete_data_and_mu.txt', self.samples)               
        
        
    def tune_covariance(self):
        
        cov_tuned = False
        
        while(cov_tuned == False):
            
            self.run_chain(tuning = True)
            
            if(self.rank == 0):
                
                if(abs(self.acceptance_rate - self.target_accept_prob) > self.accept_prob_tol):
                    
                    self.covariance = self.covariance * self.acceptance_rate / self.target_acceptance_rate
                    
                else:
                    
                    cov_tuned = True
                    
                print(self.acceptance_rate)
                
            cov_tuned = self.comm.bcast(cov_tuned, root = 0)
            self.covariance = self.comm.bcast(self.covariance, root = 0)

metropolis_sampling()
                
        



        

