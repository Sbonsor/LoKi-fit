#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:13:06 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
from dimensional_data_generation import dimensional_data_generation
import matplotlib.pyplot as plt
from scipy.special import gammainc,gamma,logsumexp
from scipy.integrate import solve_ivp,simps
import pickle
import csv
import pandas as pd


def log_likelihood(theta, rs, vs, r1, r2, model, Mhat):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    
    ### Calculate dimensional scales.
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
    M_scale = M/Mhat
    
    ### Caclulate individual star likelihoods        
    rhats = rs/rK
    vhats = vs*np.sqrt(a)
            
    psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
    Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
        
    ls = (Ae)/M * (np.exp(-Ehats) - 1)
    
    log_ls = np.log(ls)
    
    ### Calculate contribution due to mass enclosed in region.
    Mass_profile = model.M_r *M_scale
    r = model.rhat * rK
    mass_diff = np.interp(r2, xp = r, fp = Mass_profile) - np.interp(r1, xp = r, fp = Mass_profile)
    
    return np.sum(log_ls)

def r_v(data):
    
    rs = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    vs = np.sqrt(data[:,3]**2 + data[:,4]**2 + data[:,5]**2)
    
    return rs, vs

def radii_separations(Psi, rK):
    
    model = LoKi(0, 1e-6, Psi)
    r2 = np.interp(0.8*model.M_hat, model.M_r,model.rhat) * rK
    rt = model.rhat[-1]*rK
    
    return [1e-6, rK, r2, rt]

def draw_r_v_line(ax, theta, color, lwidth = 0.2):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    model = LoKi(0, 1e-6, Psi, pot_only = True)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    
    r = np.linspace(1e-6, model.rhat[-1], 1000) * rK
    psi = np.interp(r/rK, model.rhat, model.psi)
    v = np.sqrt(2*psi)/np.sqrt(a)
        
    #ax.plot(r, v, c = color, linewidth = lwidth)
    
    return r, v

def precalculate_models(Psi_axis):
        
    models = {}
        
    for Psi in Psi_axis:
        model = LoKi(0, 1e-6, Psi, pot_only = False)
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        models[str(Psi)] = {'model' : model , 'Mhat' : Mhat}    
        print(Psi)
                        
    return models

def obtain_allowed_region(PSIS, RKs, Ms, rs, vs, models, r1, r2):
    
    allowed_point = []
    log_ls = []
    
    for i in range(len(PSIS)):
        Psi = PSIS[i]
        rK = RKs[i]
        M = Ms[i]
            
        model = models[str(Psi)]['model']
        Mhat = models[str(Psi)]['Mhat']
            
        a = (9 * rK * Mhat)/(4*np.pi*G*M)
            
        ### Caclulate individual star likelihoods        
        rhats = rs/rK
        vhats = vs*np.sqrt(a)
    
        vhat_rhats = np.interp(rhats, model.rhat, np.sqrt(2*model.psi))
        
        point_comparisons = vhats <=  vhat_rhats
        
        if False in point_comparisons:
            allowed_point.append(False)
            log_l = -np.inf
            log_ls.append(log_l)
                      
        else:
            allowed_point.append(True)
            log_l = log_likelihood( [M, rK, Psi],  rs, vs, r1, r2, model, Mhat)
            log_ls.append(log_l)
      
            
    
    return np.array(allowed_point), np.array(log_ls)

# def new_samples(nsamp, M, rK, Psi, r1, r2):
    
#     sampling = dimensional_data_generation(nsamp, M, rK, Psi, 0, 1e-6, save = False, validate = False)
    
#     data = np.zeros((nsamp,6))
#     data[:, 0] = sampling.x
#     data[:, 1] = sampling.y
#     data[:, 2] = sampling.z
#     data[:, 3] = sampling.vx
#     data[:, 4] = sampling.vy
#     data[:, 5] = sampling.vz
    
#     rs, vs = r_v(data)
    
#     in_region = (rs<r2)&(rs>r1)
#     rs, vs = rs[in_region], vs[in_region]
    
#     log_r_v_matrix = np.log(np.array([rs,vs]).T)
    
#     return rs, vs, log_r_v_matrix

def new_samples(nsamp, M, rK, Psi, r1, r2):
    
    rs = np.array([])
    vs = np.array([])
    
    while (len(rs) < nsamp):
        
        sampling = dimensional_data_generation(nsamp, M, rK, Psi, 0, 1e-6, save = False, validate = False)
        
        rs_add = np.sqrt(sampling.x**2 + sampling.y**2 + sampling.z**2)
        vs_add = np.sqrt(sampling.vx**2 + sampling.vy**2 + sampling.vz**2)
        
        in_region = (rs_add<r2)&(rs_add>r1)
        rs_add, vs_add = rs_add[in_region], vs_add[in_region]
        
        rs = np.concatenate((rs, rs_add))
        vs = np.concatenate((vs, vs_add))
        
    rs = rs[0:nsamp]
    vs = vs[0:nsamp]
        
    log_r_v_matrix = np.log(np.array([rs,vs]).T)
    
    return rs, vs, log_r_v_matrix

def single_trial(M_true, rK_true, Psi_true, N_stars, models, region, M_axis, rK_axis, Psi_axis, plot = False, add_samp = False, rs_add = None, vs_add = None, log_ls_add = None):
    
    def integral_3d(integrand, rK_axis, M_axis, Psi_axis):
        
        integrand = np.reshape(integrand, (len(rK_axis), len(M_axis), len(Psi_axis))) 
        
        first_integral = simps(y = integrand, x = rK_axis, axis = 0)
        second_integral = simps(y = first_integral, x = M_axis, axis = 0)
        final_integral = simps(y = second_integral, x = Psi_axis, axis = 0)
        
        return final_integral

    def calculate_mean(Ms, RKs, PSIS, posterior):
        
        theta = [Ms, RKs, PSIS]
        
        theta_bar = np.zeros(3)
        for i in range(3):
            integrand = posterior * theta[i]
            
            theta_bar[i] = integral_3d(integrand, rK_axis, M_axis, Psi_axis)
            
        return theta_bar

    def calculate_covariance(Ms, RKs, PSIS, posterior, theta_bar):
        
        theta = [Ms, RKs, PSIS]
        
        covariance = np.zeros((3,3))

        for i in range(3):
            for j in range(i+1):
                
                integrand = (theta[i] - theta_bar[i]) * (theta[j] - theta_bar[j]) * posterior
                
                covariance[i][j] = covariance[j][i] = integral_3d(integrand, rK_axis, M_axis, Psi_axis)
                
        return covariance

    def calculate_posterior(Ms, RKs, PSIS, log_ls):
        
        ### Calculating normalisation of posterior:
        integrand = np.zeros(len(PSIS))
        ln_l_max = np.max(log_ls)

        log_diff = log_ls - ln_l_max
        integrand = np.exp(log_diff)

        integral = integral_3d(integrand, rK_axis, M_axis, Psi_axis)

        log_Z = ln_l_max + np.log(integral)

        log_posterior = log_ls - log_Z

        posterior = np.exp(log_posterior)
        
        return posterior, log_posterior
    
    ### Obtain radial boundaries of regions
    region_boundaries = radii_separations(Psi_true, rK_true)

    if region == 3:
        r1 = region_boundaries[0]
        r2 = region_boundaries[-1]
    else:
        r1 = region_boundaries[region]
        r2 = region_boundaries[region+1]
    
    ### Grid parameter space
    X,Y,Z = np.meshgrid(M_axis, rK_axis, Psi_axis)

    Ms = X.flatten()
    RKs = Y.flatten()
    PSIS = Z.flatten()

    ### Sample new stars
    rs,vs,_ = new_samples(N_stars, M_true, rK_true, Psi_true, r1, r2)

    ### Calculate allowed region and log-likelihood of the sampled set.
    allowed_region, log_ls = obtain_allowed_region(PSIS, RKs, Ms, rs, vs, models, r1, r2)
    
    ### Add another set of stars if requested
    if add_samp:
        
        rs = np.concatenate((rs, rs_add))
        vs = np.concatenate((vs, vs_add))
        log_ls = log_ls + log_ls_add
        
        
    posterior, log_posterior = calculate_posterior(Ms, RKs, PSIS, log_ls)

    theta_bar = calculate_mean(Ms, RKs, PSIS, posterior)
    
    covariance = calculate_covariance(Ms, RKs, PSIS, posterior, theta_bar)
    
    mode_idx = np.where(log_posterior == np.max(log_posterior))[0][0]
    mode = np.array([Ms[mode_idx], RKs[mode_idx], PSIS[mode_idx]])
        
    if plot:
        fig2 = plt.figure()
        ax2 = plt.axes(projection="3d")
        p = ax2.scatter3D(Ms[allowed_region], RKs[allowed_region], PSIS[allowed_region], c=log_posterior[allowed_region], alpha=0.1, marker='.')
        ax2.scatter3D(M_true, rK_true, Psi_true, c = 'r', marker = 'x', alpha = 1, label = 'True parameters')
        ax2.scatter(mode[0], mode[1], mode[2], marker = 'x', c = 'b', label = 'MAP')
        ax2.scatter(theta_bar[0], theta_bar[1], theta_bar[2], marker = 'x', c = 'k', label = 'Mean')
        fig2.colorbar(p)
        ax2.set_xlabel('M')
        ax2.set_ylabel('$r_K$')
        ax2.set_zlabel('$\\Psi$')
        ax2.legend()
    
    return rs, vs, allowed_region, log_ls, posterior, mode, theta_bar, covariance

### List of cluster to fit.
clusters = ['MW-NGC104', 'MW-PAL5','MW-NGC6229','MW-PAL3', 'MW-NGC7492', 'MW-NGC6809', 'MW-NGC6553', 'MW-NGC6402', 'MW-NGC5139', 'MW-NGC2419','MW-NGC1904']

### Granularity of axes
npoints_axis = 100

### File containing cluster parameters
cluster_params_file = 'Data/asu.tsv'

### Read parameter file
cluster_params = pd.read_csv (cluster_params_file, sep = '\t',comment='#')

### Select King models
cluster_params = cluster_params[cluster_params['Mod'] == 'K ']

### Eliminate leading or trailing whitespace from strings
cluster_params = cluster_params.applymap(lambda x: x.strip() if isinstance(x, str) else x)

### Physical constants.
G = 4.3009e-3

results_dict = {}
results_dict['npoints'] = npoints_axis

for cluster in clusters:
    
    results_dict[cluster] = {}    
    
    ### Read in parameters
    true_params = cluster_params[cluster_params['Cluster'] == cluster]
    
    ### True model parameters.
    M_true = 10 ** true_params['Mtot'].astype(float).to_numpy()[0]
    rK_true = true_params['r0pc'].astype(float).to_numpy()[0]
    Psi_true = true_params['W0/gamma'].astype(float).to_numpy()[0]
    
    ### Number of stars to use in the sample.
    N_stars = 10000
    
    ### Parameter space gridding using quoted errors
    
    M_max = 10**(true_params['Mtot'].astype(float).to_numpy()[0] + 2*true_params['E_Mtot'].astype(float).to_numpy()[0])
    M_min = 10**(true_params['Mtot'].astype(float).to_numpy()[0] - 2*true_params['e_Mtot'].astype(float).to_numpy()[0])
    rK_min = max(true_params['r0pc'].astype(float).to_numpy()[0] - 5*true_params['e_r0pc'].astype(float).to_numpy()[0], 0.01)
    rK_max = true_params['r0pc'].astype(float).to_numpy()[0] + 5*true_params['E_r0pc'].astype(float).to_numpy()[0]
    Psi_min = max(true_params['W0/gamma'].astype(float).to_numpy()[0] - 5*true_params['e_W0/gamma'].astype(float).to_numpy()[0], 0.1)
    Psi_max = true_params['W0/gamma'].astype(float).to_numpy()[0] + 5*true_params['E_W0/gamma'].astype(float).to_numpy()[0]
    
    M_axis = np.linspace(M_min, M_max, npoints_axis)
    Psi_axis = np.linspace(Psi_min, Psi_max, npoints_axis)
    rK_axis = np.linspace(rK_min, rK_max, npoints_axis)

    models = precalculate_models(Psi_axis)

    ### Select base region to examine, and obtain the radial bounds of that region (don't change).
    region = 3
    
    ### Perform fitting
    rs, vs, allowed_region, log_ls, posterior, mode, theta_bar, cov = single_trial(M_true, rK_true, Psi_true, N_stars, models, region, M_axis, rK_axis, Psi_axis, plot = True)
    
    results_dict[cluster]['samples'] = [rs, vs]
    results_dict[cluster]['mean'] = theta_bar
    results_dict[cluster]['covariance'] = cov
    results_dict[cluster]['MAP'] = mode
    results_dict[cluster]['allowed_region'] = allowed_region
    results_dict[cluster]['log_ls'] = log_ls
    results_dict[cluster]['posterior'] = posterior

fname = 'multi_cluster_run.pkl'        
with open(fname, 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    












































# ### Obtain region boundaries

# region_boundaries = radii_separations(Psi, rK)

# if region == 3:
#     r1 = region_boundaries[0]
#     r2 = region_boundaries[-1]
# else:
#     r1 = region_boundaries[region]
#     r2 = region_boundaries[region+1]

# ### Gridding parameter space.
# M_axis = np.linspace(M_min,M_max, npoints_axis)
# rK_axis = np.linspace(rK_min, rK_max, npoints_axis)
# Psi_axis = np.linspace(Psi_min, Psi_max, npoints_axis)

# X,Y,Z = np.meshgrid(M_axis, rK_axis, Psi_axis)

# Ms = X.flatten()
# RKs = Y.flatten()
# PSIS = Z.flatten()

# ### Sample stars
# rs,vs,_ = new_samples(N_stars, M, rK, Psi, r1, r2)

# ### Pre-calculate LoKi models
# models = precalculate_models(Psi_axis)

# ### Calculate allowed region
# allowed_region, log_ls = obtain_allowed_region(PSIS, RKs, Ms, rs, vs, models)

# mode_idx = np.where(log_ls == np.max(log_ls))[0][0]
# mode = np.array([Ms[mode_idx], RKs[mode_idx], PSIS[mode_idx]])

# fig1 = plt.figure()
# ax1 = plt.axes(projection="3d")
# p = ax1.scatter3D(Ms[allowed_region], RKs[allowed_region], PSIS[allowed_region], c=log_ls[allowed_region], alpha=0.1, marker='.')
# ax1.scatter3D(M, rK, Psi, c = 'r', marker = 'x', alpha = 1)
# ax1.scatter(mode[0], mode[1], mode[2], marker = 'x', c = 'b', label = 'Mode')
# fig1.colorbar(p)
# ax1.set_xlabel('M')
# ax1.set_ylabel('$r_K$')
# ax1.set_zlabel('$\\Psi$')
# ax1.legend()


# ### Calculating normalisation:
# integrand = np.zeros(len(PSIS))
# ln_l_max = np.max(log_ls)

# log_diff = log_ls - ln_l_max
# integrand = np.exp(log_diff)

# integral = integral_3d(integrand, rK_axis, M_axis, Psi_axis)

# log_Z = ln_l_max + np.log(integral)

# log_posterior = log_ls - log_Z

# posterior = np.exp(log_posterior)

# theta = [Ms, RKs, PSIS]
# theta_bar = np.zeros(3)
# for i in range(3):
#     integrand = posterior * theta[i]
    
#     theta_bar[i] = integral_3d(integrand, rK_axis, M_axis, Psi_axis)

# fig2 = plt.figure()
# ax2 = plt.axes(projection="3d")
# p = ax2.scatter3D(Ms[allowed_region], RKs[allowed_region], PSIS[allowed_region], c=log_posterior[allowed_region], alpha=0.1, marker='.')
# ax2.scatter3D(M, rK, Psi, c = 'r', marker = 'x', alpha = 1)
# mode_idx = np.where(log_posterior == np.max(log_posterior))[0][0]
# mode = np.array([Ms[mode_idx], RKs[mode_idx], PSIS[mode_idx]])
# ax2.scatter(mode[0], mode[1], mode[2], marker = 'x', c = 'b', label = 'MAP')
# ax2.scatter(theta_bar[0], theta_bar[1], theta_bar[2], marker = 'x', c = 'k', label = 'Mean')
# fig2.colorbar(p)
# ax2.set_xlabel('M')
# ax2.set_ylabel('$r_K$')
# ax2.set_zlabel('$\\Psi$')
# ax2.legend()

# covariance = np.zeros((3,3))

# for i in range(3):
#     for j in range(i+1):
        
#         integrand = (theta[i] - theta_bar[i]) * (theta[j] - theta_bar[j]) * posterior
        
#         covariance[i][j] = covariance[j][i] = integral_3d(integrand, rK_axis, M_axis, Psi_axis)
        




    

    
    














