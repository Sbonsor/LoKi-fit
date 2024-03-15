#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:38:00 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import gammainc, gamma
from mpi4py import MPI

def log_likelihood(x, theta):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    mu = theta[3]
    epsilon = theta[4]
    G = 4.3009e-3
       
    model = LoKi(mu, epsilon, Psi, pot_only = True)
    
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
        
    x_ = x[0].copy()
    y_ = x[1].copy()
    z_ = x[2].copy()
    vx_ = x[3].copy()
    vy_ = x[4].copy()
    vz_ = x[5].copy()
          
    x_ *= 1/rK
    y_ *= 1/rK
    z_ *= 1/rK
    vx_ *= np.sqrt(a)
    vy_ *= np.sqrt(a)
    vz_ *= np.sqrt(a)
            
    rhat = np.sqrt(x_**2 + y_**2 + z_**2)
    vhat = np.sqrt(vx_**2 + vy_**2 + vz_**2)
        
    psi = np.interp(rhat, xp = model.rhat, fp = model.psi)
    Ehat = np.clip(0.5* vhat**2 - psi, a_max = 0, a_min = None)
        
    l = (Ae)/M * (np.exp(-Ehat) - 1)
    
    log_l= np.log(l)
    
    return log_l

# def log_likelihood(x, theta):
    
#     a = theta[0]
#     b = theta[1]
#     c = theta[2]
    
#     log_l = a*b*c + a**2 * b**2 * c**2
    
#     return log_l

def mixed_derivative(i, j, h, data_point, theta):
    
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
    
    y_increment = np.zeros(len(theta))
    y_increment[j] = h

    u_x2_y2 = log_likelihood(data_point, theta + 2*x_increment + 2*y_increment)
    
    u_x1_y1 = log_likelihood(data_point, theta + 1*x_increment + 1*y_increment)
    
    u_x2_2y = log_likelihood(data_point, theta + 2*x_increment - 2*y_increment)
    
    u_x1_1y = log_likelihood(data_point, theta + 1*x_increment - 1*y_increment)
    
    u_2x_y2 = log_likelihood(data_point, theta - 2*x_increment + 2*y_increment)
    
    u_1x_y1 = log_likelihood(data_point, theta - 1*x_increment + 1*y_increment)
    
    u_2x_2y = log_likelihood(data_point, theta - 2*x_increment - 2*y_increment)
    
    u_1x_1y = log_likelihood(data_point, theta - 1*x_increment - 1*y_increment)
    
    
    #mixed_derivative = (-u_x2_y2 + 16*u_x1_y1 + u_x2_2y - 16*u_x1_1y + u_2x_y2 - 16*u_1x_y1 - u_2x_2y + 16*u_1x_1y )/(48*h**2)
    mixed_derivative = (u_x1_y1 - u_x1_1y - u_1x_y1 + u_1x_1y)/(4*h**2)
    
    return mixed_derivative

def second_derivative(i, h, data_point, theta):
    
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
    
    u_x = log_likelihood(data_point, theta)
    
    u_x2 = log_likelihood(data_point, theta + 2*x_increment)
    u_x1 = log_likelihood(data_point, theta + 1*x_increment)
    u_1x = log_likelihood(data_point, theta - 1*x_increment)
    u_2x = log_likelihood(data_point, theta - 2*x_increment)
    
    second_deriv = (-u_x2 + 16*u_x1 - 30*u_x + 16*u_1x - u_2x)/(12*h**2)
    
    return second_deriv

def Hessian(data_point, theta, h):
    
    hessian = np.zeros((3,3))
    
    for i in range(3):
        for j in range(i,3,1):
            
            if i == j:    
                hessian[i][j] = second_derivative(i, h, data_point, theta)
            else:
                hessian[i][j] = hessian[j][i] = mixed_derivative(i, j, h, data_point, theta)
    
    return hessian

# def observed_fisher_information(theta, data, N, h):
    
#     J = np.zeros((3,3))
    
#     for i in range(N):
        
#         data_point = data[i,:]
        
#         H = Hessian(data_point, theta, h)
        
#         J = J + H
        
#     J = J/N
    
#     return -J

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6

restart = False
# data_path = f'/home/s1984454/LoKi-fit/Data/dimensional_samples_King_M_{M}_rK_{rK}_Psi_{Psi}_mu_{mu}_epsilon_{epsilon}_N_1000000.txt'
# save_file = f'/home/s1984454/LoKi-fit/Data/fisher_evaluations_King_M_{M}_rK_{rK}_Psi_{Psi}_mu_{mu}_epsilon_{epsilon}_N_1000000_processor_{rank}.txt'

data_path = '/home/s1984454/LoKi-fit/Data/dimensional_samples_Base_sample_set.txt'
save_file = f'/home/s1984454/LoKi-fit/Data/fisher_evaluations_Base_sample_set_{rank}.txt'

theta = np.array([M, rK, Psi, mu, epsilon])
h = 1e-4

# Check for how much has already been calculated and set the starting index appropriately.
if (restart == True):
    
    existing_data = np.loadtxt(save_file)
    start_index = len(existing_data) - 1

#Otherwise create a new empty file.    
else:

    open(save_file, 'w').close() 
    start_index = 0

if (rank == 0):
    
    data = np.loadtxt(data_path) #Load data
    data = np.array_split(data,size) #Generate list with subsets of the data to go to each processor
    
else:
    
    data = None
    
data = comm.scatter(data,root = 0) # Distribute each sub-array to it's processor.

for i in range(start_index, len(data)):
    print(i)
    
    data_row = data[i,:]
    
    minus_H = -Hessian(data_row, theta, h)
    
    write_row = np.reshape(np.append(data_row, minus_H), (1,len(data_row)+9))
    
    with open(save_file, 'ab') as f:
        np.savetxt(f, write_row, delimiter= ' ')
    
    
    

        
    

































