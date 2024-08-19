#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:31:50 2024

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
from dimensional_data_generation import dimensional_data_generation
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd
plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 16

G = 4.3009e-3


def radii_separations(Psi, rK):
    
    model = LoKi(0, 1e-6, Psi)
    r2 = np.interp(0.8*model.M_hat, model.M_r,model.rhat) * rK
    rt = model.rhat[-1]*rK
    
    return [1e-6, rK, r2, rt]

def log_likelihood(theta, rs, vs, model, Mhat, region_boundaries, completeness_profile):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    
    ### Calculate dimensional scales.
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    Ae = (3 * a**(3/2) * M)/(8 * np.sqrt(2) * np.pi * model.density(Psi) * rK**3 * Mhat)
    
    ### Caclulate individual star likelihoods        
    rhats = rs/rK
    vhats = vs*np.sqrt(a)
            
    psis = np.interp(rhats, xp = model.rhat, fp = model.psi)
    Ehats = np.clip(0.5* vhats**2 - psis, a_max = 0, a_min = None)
        
    ls = (Ae)/M * (np.exp(-Ehats) - 1)
    
    ### Calculate completeness effect.
    
    region_0_idx = (rs>=region_boundaries[0])&(rs<region_boundaries[1])
    region_1_idx = (rs>=region_boundaries[1])&(rs<region_boundaries[2])
    region_2_idx = (rs>=region_boundaries[2])&(rs<region_boundaries[3])
    
    ls[region_0_idx] = ls[region_0_idx] * completeness_profile[0]
    ls[region_1_idx] = ls[region_1_idx] * completeness_profile[1]
    ls[region_2_idx] = ls[region_2_idx] * completeness_profile[2]
    
    log_ls = np.log(ls)
    
    return np.sum(log_ls)

def precalculate_models(Psi_axis):
        
    models = {}
        
    for Psi in Psi_axis:
        model = LoKi(0, 1e-6, Psi, pot_only = False)
        Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
        models[str(Psi)] = {'model' : model , 'Mhat' : Mhat}    
        print(Psi)
                        
    return models

def new_samples(nsamp, M, rK, Psi):
    
    rs = np.array([])
    vs = np.array([])
    
    while (len(rs) < nsamp):
        
        sampling = dimensional_data_generation(nsamp, M, rK, Psi, 0, 1e-6, save = False, validate = False)
        
        rs = np.sqrt(sampling.x**2 + sampling.y**2 + sampling.z**2)
        vs = np.sqrt(sampling.vx**2 + sampling.vy**2 + sampling.vz**2)
    
    return rs, vs

def obtain_allowed_region(PSIS, RKs, Ms, rs, vs, models, region_boundaries, completeness_profile):
    
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
            log_l = log_likelihood( [M, rK, Psi],  rs, vs, model, Mhat, region_boundaries, completeness_profile)
            log_ls.append(log_l)
        
        if(i%1000==0):
            print(i)
    
    return np.array(allowed_point), np.array(log_ls)

def single_trial(rs, vs, models, M_axis, rK_axis, Psi_axis, region_boundaries, completeness_profile, plot = False):
    
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
        
    ### Grid parameter space
    X,Y,Z = np.meshgrid(M_axis, rK_axis, Psi_axis)

    Ms = X.flatten()
    RKs = Y.flatten()
    PSIS = Z.flatten()

    ### Calculate allowed region and log-likelihood of the sampled set.
    allowed_region, log_ls = obtain_allowed_region(PSIS, RKs, Ms, rs, vs, models, region_boundaries, completeness_profile)
    
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
    
    return allowed_region, log_ls, posterior, mode, theta_bar, covariance, log_posterior
        
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

def select_subset(rs_full, vs_full, region_boundaries, completeness_profile):
    
    region = np.zeros(len(rs_full))
    
    region_0_idx = (rs_full>=region_boundaries[0])&(rs_full<region_boundaries[1])
    region_1_idx = (rs_full>=region_boundaries[1])&(rs_full<region_boundaries[2])
    region_2_idx = (rs_full>=region_boundaries[2])&(rs_full<region_boundaries[3])
    
    region[region_0_idx] = 0
    region[region_1_idx] = 1
    region[region_2_idx] = 2
    
    rs = []
    vs = []
    
    for i in range(len(rs_full)):
        acceptance_prob = completeness_profile[int(region[i])]
        
        alpha = np.random.uniform()
        
        if (alpha <= acceptance_prob):
            
            rs.append(rs_full[i])
            vs.append(vs_full[i])
    
    return np.array(rs), np.array(vs)

n_stars = 100000
npoints_axis = 100
Psi = 5
M = 500
rK = 1.2

completeness_profile = [0.1, 0.2, 0.1]

### True model parameters.
M_true = M
rK_true = rK
Psi_true = Psi

M_max = 650
M_min = 475
rK_min = 0.01
rK_max = 1.75
Psi_min = 4
Psi_max = 10

M_axis = np.linspace(M_min, M_max, npoints_axis)
Psi_axis = np.linspace(Psi_min, Psi_max, npoints_axis)
rK_axis = np.linspace(rK_min, rK_max, npoints_axis)

region_boundaries = radii_separations(Psi_true, rK_true)

models = precalculate_models(Psi_axis)

rs_full, vs_full = new_samples(n_stars, M_true, rK_true, Psi_true)

rs_subset, vs_subset = select_subset(rs_full, vs_full, region_boundaries, completeness_profile)

allowed_region, log_ls, posterior, mode, theta_bar, covariance, log_posterior = single_trial(rs_subset, vs_subset, models, M_axis, rK_axis, Psi_axis, region_boundaries, completeness_profile, plot = True)



