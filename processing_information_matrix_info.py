#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:38:59 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import simps
from scipy.special import gammainc, gamma



def combine_data(base_file, nprocessors):
    
    data = np.loadtxt(f'Data/{base_file}_0.txt')

    for processor in range(1,nprocessors):
        data_to_add = np.loadtxt(f'Data/{base_file}_{processor}.txt')

        data = np.concatenate((data, data_to_add), axis = 0)
        
    return data

def r_v_detJ(data):
    
    rs = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    vs = np.sqrt(data[:,3]**2 + data[:,4]**2 + data[:,5]**2)
    Js = data[:,6:]

    determinants = []
    matrices = []
    for i in range(len(Js)):    
        matrix = np.reshape(Js[i,:], (3,3))
        determinants.append(np.linalg.det(matrix))
        matrices.append(matrix)
    return rs, vs, np.array(determinants), matrices

def convergence_in_N(matrices):
    Ns = np.logspace(1,np.log10(len(matrices)), 20,  dtype = int)
    
    determinants = []
    for N in Ns:
        J = sum(matrices[0:N])/N
        det_J= np.linalg.det(J)
        
        determinants.append(det_J)
    
    fig,ax = plt.subplots(1,1)
    ax.scatter(Ns, determinants, marker = 'x')
    ax.set_xlabel('N')
    ax.set_ylabel('det(J)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    return Ns, determinants

def generate_figures():
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(x=rs, y= determinants_i, marker = 'x')
    ax.set_xlabel('r')
    ax.set_ylabel('det(J_i)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(x=rs, y=vs, c=np.log(determinants_i), cmap = 'coolwarm', marker = 'x', s = 1)
    ax.set_xlabel('r')
    ax.set_ylabel('v')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    model = LoKi(mu, epsilon, Psi, pot_only = True)
    G = 4.3009e-3
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    psi = np.interp(rs/rK, model.rhat, model.psi)
    v = np.sqrt(2*psi)/np.sqrt(a)
    
    idxs = rs.argsort()
    ax.plot(rs[idxs], v[idxs], linewidth = 0.1, c = 'k')
    #fig.colorbar(np.log(determinants_i), ax = ax)
    
    return 1

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


base_file = 'fisher_evaluations_King_M_500_rK_1.2_Psi_5_mu_0_epsilon_1e-06_N_1000000_processor'
nprocessors = 10
M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6
theta = np.array([M, rK, Psi, mu, epsilon])
h = 1e-4

data = combine_data(base_file, nprocessors)
rs, vs, determinants_i, matrices = r_v_detJ(data)

Ns, determinants = convergence_in_N(matrices)

generate_figures()


### Most extreme point removal

max_idx = np.where(determinants_i == max(determinants_i))[0][0]
#max_idx = np.where(rs == max(rs))[0][0]
print(max(determinants_i))
max_data = data_point = data[max_idx,:]
max_hessian = -Hessian(max_data, theta, h)
max_hessian_det = np.linalg.det(max_hessian)
print(max_hessian_det)



############ Evaluation of log likelihood at point
G = 4.3009e-3
x = max_data   
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

data = np.delete(data, max_idx , axis = 0)
rs, vs, determinants_i, matrices = r_v_detJ(data)

Ns, determinants = convergence_in_N(matrices)
generate_figures()








