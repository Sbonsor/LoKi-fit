#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:19:04 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
from dimensional_data_generation import dimensional_data_generation
import matplotlib.pyplot as plt
from scipy.special import gammainc,gamma,logsumexp
from scipy.integrate import solve_ivp,simps
import pickle
plt.rc('text', usetex=True)

### Loading results files.
#fname = 'M_1122018.4543019629_rK_0.51_8.6_validation_run_no_region.pkl'
fname = 'M_1122018.4543019629_rK_0.51_8.6_validation_run.pkl'    
with open(fname, 'rb') as handle:
     val_dict = pickle.load(handle)

#fname = 'M_1122018.4543019629_rK_0.51_8.6_mean_var_run_no_region.pkl'     
fname = 'M_1122018.4543019629_rK_0.51_8.6_mean_var_run.pkl'    
with open(fname, 'rb') as handle:
     results_dict = pickle.load(handle)

### Reading in parameters
Psi = results_dict['parameters']['Psi']
rK = results_dict['parameters']['rK']
M = results_dict['parameters']['M']
M_min = results_dict['parameters']['M_min']
M_max = results_dict['parameters']['M_max']
rK_min = results_dict['parameters']['rK_min']
rK_max = results_dict['parameters']['rK_max']
Psi_min = results_dict['parameters']['Psi_min']
Psi_max = results_dict['parameters']['Psi_max']

npoints_axis = 100
M_axis = np.linspace(M_min, M_max, npoints_axis)
Psi_axis = np.linspace(Psi_min, Psi_max, npoints_axis)
rK_axis = np.linspace(rK_min, rK_max, npoints_axis)
X,Y,Z = np.meshgrid(M_axis, rK_axis, Psi_axis)
Ms = X.flatten()
RKs = Y.flatten()
PSIS = Z.flatten()

### Validation figs
M_vars = []
rK_vars = []
Psi_vars = []
Ns = []

for key in val_dict.keys():
    
    if (key != 'parameters'):
        
        Ns.append(int(key))
        cov = val_dict[key]['covariance']
        
        M_vars.append(cov[0][0])
        rK_vars.append(cov[1][1])
        Psi_vars.append(cov[2][2])

fig, ax = plt.subplots(1,1)
ax.plot(Ns, M_vars, marker = 'x')
ax.set_xlabel('N')
ax.set_ylabel('$\\sigma^2_M$')

fig, ax = plt.subplots(1,1)
ax.plot(Ns, rK_vars, marker = 'x')
ax.set_xlabel('N')
ax.set_ylabel('$\\sigma^2_{r_K}$')

fig, ax = plt.subplots(1,1)
ax.plot(Ns, Psi_vars, marker = 'x')
ax.set_xlabel('N')
ax.set_ylabel('$\\sigma^2_{\\Psi}$')


colors = ['c', 'm', 'y', 'purple']
true_params = np.array([M,rK,Psi])

region = 3
covariances = []
changes_in_trace = []
for i in range(1):
#for i in range(len(results_dict)-1):
    
    trial_data = results_dict[f'trial_{i}']
    
    initial_mean = trial_data['initial_mean']
    initial_MAP = trial_data['initial_MAP']
    initial_cov = trial_data['initial_covariance']
    initial_trace = np.trace(initial_cov)
    
    covariances.append(initial_cov)
    initial_mean_error = initial_mean - true_params
    
    print(f'Trial {i} initial estimate M: mean  = {initial_mean[0]}, variance = {initial_cov[0][0]}' )
    print(f'Trial {i} initial estimate rK: mean  = {initial_mean[1]}, variance = {initial_cov[1][1]}' )
    print(f'Trial {i} initial estimate Psi: mean  = {initial_mean[2]}, variance = {initial_cov[2][2]}' )
    print(f'Trial {i} initial covariance trace = {initial_trace}')
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(M, rK, Psi , c = 'r', alpha=1, marker='x', label = 'True parameters')
    ax.scatter3D(initial_mean[0], initial_mean[1] , initial_mean[2] , c = 'k', marker = 'x', alpha = 1, label = 'Starting mean')
    ax.scatter3D(initial_MAP[0], initial_MAP[1] , initial_MAP[2] , c = 'b', marker = 'x', alpha = 1, label = 'Starting MAP')
        
    region_mean = trial_data[f'region_{region}']['theta_bar']
    region_MAP = trial_data[f'region_{region}']['MAP']
    region_cov = trial_data[f'region_{region}']['covariance']
    region_trace = np.trace(region_cov)
    
    change_in_trace = initial_trace - region_trace
    changes_in_trace.append(change_in_trace)
    
    print(f'Trial {i} region {region} estimate M: mean  = {region_mean[0]}, variance = {region_cov[0][0]}' )
    print(f'Trial {i} region {region} estimate rK: mean  = {region_mean[1]}, variance = {region_cov[1][1]}' )
    print(f'Trial {i} region {region} estimate Psi: mean  = {region_mean[2]}, variance = {region_cov[2][2]}' )
    print(f'Trial {i} region {region} covariance trace = {np.trace(region_cov)}')
    
    print(f'Trial {i} region {region} change in trace = {change_in_trace}')
    
    covariances.append(region_cov)
    
    ax.scatter3D(region_mean[0], region_mean[1] , region_mean[2] , c = colors[region], marker = 'x', alpha = 1, label = f'Region {region} mean')
    ax.scatter3D(region_MAP[0], region_MAP[1], region_MAP[2] , c = colors[region], marker = 'x', alpha = 1)
    
    region_mean_error = region_mean - true_params
        
    ax.set_xlabel('M')
    ax.set_ylabel('$r_K$')
    ax.set_zlabel('$\\Psi$')
    ax.set_xlim((M_min, M_max))
    ax.set_ylim((rK_min, rK_max))
    ax.set_zlim((Psi_min, Psi_max))
    ax.legend()
    plt.show()
    
# ##### Examining change in likelihood function, why is the result displaced?
    initial_log_ls = trial_data['initial_log_ls']
    final_log_ls = trial_data[f'region_{region}']['log_ls']
    allowed_region = trial_data[f'region_{region}']['allowed_region']
    
    allowed_region = [True]* len(final_log_ls)    
    for j in range(len(final_log_ls)):
        if(final_log_ls[j] == -np.inf):
            allowed_region[j] = False
        
    log_ls_change = final_log_ls[allowed_region] - initial_log_ls[allowed_region]
    
    fig2 = plt.figure()
    ax2 = plt.axes(projection="3d")
    p = ax2.scatter3D(Ms[allowed_region], RKs[allowed_region], PSIS[allowed_region], c=log_ls_change, alpha=0.1, marker='.')
    ax2.scatter3D(M, rK, Psi, c = 'r', marker = 'x', alpha = 1, label = 'True parameters')
    # ax2.scatter(mode[0], mode[1], mode[2], marker = 'x', c = 'b', label = 'MAP')
    # ax2.scatter(theta_bar[0], theta_bar[1], theta_bar[2], marker = 'x', c = 'k', label = 'Mean')
    fig2.colorbar(p)
    ax2.set_xlabel('M')
    ax2.set_ylabel('$r_K$')
    ax2.set_zlabel('$\\Psi$')
    ax2.legend()
    ax2.set_title('Change in log-likelihood')
    
    fig3 = plt.figure()
    ax3 = plt.axes(projection="3d")
    p = ax3.scatter3D(Ms[allowed_region], RKs[allowed_region], PSIS[allowed_region], c=final_log_ls[allowed_region], alpha=0.1, marker='.')
    ax3.scatter3D(M, rK, Psi, c = 'r', marker = 'x', alpha = 1, label = 'True parameters')
    ax3.scatter(region_mean[0], region_mean[1], region_mean[2], marker = 'x', c = 'k', label = 'Mean')
    fig3.colorbar(p)
    ax3.set_xlabel('M')
    ax3.set_ylabel('$r_K$')
    ax3.set_zlabel('$\\Psi$')
    ax3.legend()
    ax3.set_title('log-likelihood')


    
    



