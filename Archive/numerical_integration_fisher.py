#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:40:24 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import gammainc, gamma
from mpi4py import MPI

def log_likelihood(r, v, theta):
    
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
        
    rhat = r/rK 
    vhat = v*np.sqrt(a)
        
    psi = np.interp(rhat, xp = model.rhat, fp = model.psi)
    Ehat = np.clip(0.5* vhat**2 - psi, a_max = 0, a_min = None)
        
    l = (Ae)/M * (np.exp(-Ehat) - 1)
    
    log_l= np.log(l)
    
    return log_l


def mixed_derivative(i, j, h, r, v, theta):
    
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
    
    y_increment = np.zeros(len(theta))
    y_increment[j] = h

    u_x2_y2 = log_likelihood(r, v, theta + 2*x_increment + 2*y_increment)
    
    u_x1_y1 = log_likelihood(r, v, theta + 1*x_increment + 1*y_increment)
    
    u_x2_2y = log_likelihood(r, v, theta + 2*x_increment - 2*y_increment)
    
    u_x1_1y = log_likelihood(r, v, theta + 1*x_increment - 1*y_increment)
    
    u_2x_y2 = log_likelihood(r, v, theta - 2*x_increment + 2*y_increment)
    
    u_1x_y1 = log_likelihood(r, v, theta - 1*x_increment + 1*y_increment)
    
    u_2x_2y = log_likelihood(r, v, theta - 2*x_increment - 2*y_increment)
    
    u_1x_1y = log_likelihood(r, v, theta - 1*x_increment - 1*y_increment)
    
    
    #mixed_derivative = (-u_x2_y2 + 16*u_x1_y1 + u_x2_2y - 16*u_x1_1y + u_2x_y2 - 16*u_1x_y1 - u_2x_2y + 16*u_1x_1y )/(48*h**2)
    mixed_derivative = (u_x1_y1 - u_x1_1y - u_1x_y1 + u_1x_1y)/(4*h**2)
    
    return mixed_derivative

def second_derivative(i, h, r, v, theta):
    
    x_increment = np.zeros(len(theta))
    x_increment[i] = h
    
    u_x = log_likelihood(r, v, theta)
    
    u_x2 = log_likelihood(r, v, theta + 2*x_increment)
    u_x1 = log_likelihood(r, v, theta + 1*x_increment)
    u_1x = log_likelihood(r, v, theta - 1*x_increment)
    u_2x = log_likelihood(r, v, theta - 2*x_increment)
    
    second_deriv = (-u_x2 + 16*u_x1 - 30*u_x + 16*u_1x - u_2x)/(12*h**2)
    
    return second_deriv

n_r = 10
n_v = 20

M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6
theta = [M, rK, Psi, mu, epsilon]
G = 4.3009e-3
h = 1e-4
   
model = LoKi(mu, epsilon, Psi, pot_only = True)

Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
a = (9 * rK * Mhat)/(4*np.pi*G*M)
Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)

rs = np.zeros((n_v,n_r))
vs = np.zeros((n_v,n_r))

for j in range(n_v):
    rs[j,:] = np.linspace(epsilon, model.rhat[-1], n_r)*rK

for i in range(n_r):
    
    psi = np.interp(rs[0, i]/rK, xp = model.rhat, fp = model.psi)
    vhat_max = np.sqrt(2*psi)
    v_max = vhat_max / np.sqrt(a)
    
    vs[:, i] = np.linspace(0, v_max, n_v)
    
information_matrix = np.zeros((3,3))

for i in range(3):
    for j in range(i,3,1):
        integrand = np.zeros((n_v,n_r))
        
        for i1 in range(n_v):
            for j1 in range(n_r):
                
                    if i == j:    
                        integrand[i1,j1] = second_derivative(i, h, rs[i1,j1], vs[i1,j1], theta)
                    else:
                         integrand[i1,j1] = mixed_derivative(i, j, h, rs[i1,j1], vs[i1,j1], theta)
                    print(f'({i1},{j1})')
                    
        integrand = np.nan_to_num(integrand)
        
        integral_in_velocity = np.zeros(n_r)
        
        for idx in range(n_r):
            integral_in_velocity[idx] = simps(y = integrand[:, i], x = vs[:,i])
        
        information_matrix[i,j] = information_matrix[j,i] = simps(y = integral_in_velocity, x = rs[0,:])        
    
    