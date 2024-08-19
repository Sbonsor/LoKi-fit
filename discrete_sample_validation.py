#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:24:04 2024

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
from LoKi_samp import LoKi_samp

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
    
    model = LoKi(0, 1e-6, Psi, pot_only = False)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    
    A_hat = M/(Mhat * rho_hat(Psi) * rK**3)    
    sqrt_a = np.sqrt( 9/(4 * np.pi * G * A_hat * rho_hat(Psi) * rK**2) )
    
    rhats = rs/rK
    vhats = vs*sqrt_a
    
    return rs, vs, rhats, vhats

def r_v(data):
    
    rs = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    vs = np.sqrt(data[:,3]**2 + data[:,4]**2 + data[:,5]**2)
    
    return rs, vs

def radii_separations(Psi, rK):
    
    model = LoKi(0, 1e-6, Psi)
    r2 = np.interp(0.8*model.M_hat, model.M_r,model.rhat) * rK
    rt = model.rhat[-1]*rK
    
    return [1e-6, rK, r2, rt]

def rho_hat(psi):
    
    density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
    density = np.nan_to_num(density,copy = False)
        
    return density

def region_boundaries(region, Psi, rK):
    
    region_boundaries = radii_separations(Psi, rK)
    
    if region == 3:
        r1 = region_boundaries[0]
        r2 = region_boundaries[-1]
    else:
        r1 = region_boundaries[region]
        r2 = region_boundaries[region+1]
        
    return r1, r2

def rad_prob_density_profile(rhats, vhats):
    
    def density_normalisation(r,rho):
        integrand = r**2*rho
        return np.trapz(y = integrand, x = r)

    fig1,ax1 = plt.subplots(1,1)
    ax1.plot(model.rhat,model.rhat**2*model.rho_hat/density_normalisation(model.rhat,model.rho_hat))
    ax1.hist(rhats,density = True,bins = 50)
    ax1.set_xlabel('$\hat{r}$')
    ax1.set_ylabel('Radial probability density')
    ax1.set_title(f'N = {len(rhats)}')
    
    nks, bin_edges = np.histogram(rhats, density=False, bins = 50)        
    rks = np.zeros(len(nks))
    for i in range(len(nks)):
        rks[i] = (bin_edges[i+1]+bin_edges[i])/2
        
    drs = [bin_edges[i+1]-bin_edges[i] for i in range(len(nks))]
    rho_ks = nks/(4*np.pi*rks**2*drs)

    dr0 = drs[0]
    n0 = np.sum(rhats<dr0)
    rho_0 = n0/((4/3) * np.pi * dr0**3)

    fig2, ax2 = plt.subplots(1,1)
    ax2.scatter(x = rks, y = rho_ks/rho_0, marker = 'x')
    ax2.plot(model.rhat, model.density(model.psi)/model.density(Psi), color = 'r')
    ax2.set_yscale('log')
    ax2.set_xlabel('\\hat{r}')
    ax2.set_ylabel('$log\\hat\\rho/\\rho_0$')
    ax2.set_title(f'N = {len(rhats)}')
    
    return rks, rho_ks/rho_0, rho_ks, rho_0

G = 4.3009e-3

Psi = 6
rK = 1.2
M = 500

initial_region  = 3
add_region = 2
n_star_initial = 10000
n_star_add = 10000

model = LoKi(0, 1e-6, Psi, pot_only = False)
Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)

### Initial sample
# r1, r2 = region_boundaries(initial_region, Psi, rK)
# rs, vs, rhats, vhats = new_samples(n_star_initial, M, rK, Psi, r1, r2)
samples = LoKi_samp(model, N = n_star_initial, plot = False, scale_nbody = False)
rhats = samples.r
vhats = samples.v

xhats = samples.x
yhats = samples.y
zhats = samples.z
vxhats = samples.vx
vyhats = samples.vy
vzhats = samples.vz

rks, normalised_density, rho_ks, rho_0 = rad_prob_density_profile(rhats, vhats)

def pressure_profile(samples):
    
    rs = samples.r
    vs = samples.v

    xs = samples.x
    ys = samples.y
    zs = samples.z
    vxs = samples.vx
    vys = samples.vy
    vzs = samples.vz
    
    nks, bin_edges = np.histogram(rhats, density=False, bins = 50)        
    rks = np.zeros(len(nks))
    rks = np.zeros(len(nks))
    for i in range(len(nks)):
        rks[i] = (bin_edges[i+1]+bin_edges[i])/2
    
    drs = [bin_edges[i+1]-bin_edges[i] for i in range(len(nks))]
    rho_ks = nks/(4*np.pi*rks**2*drs)
    
    Pks = np.zeros(len(nks))
    for i in range(len(nks)):
        in_shell = (rs < bin_edges[i+1])&(rs>bin_edges[i])
        
        v_rsi = (xs[in_shell] * vxs[in_shell] + ys[in_shell] * vys[in_shell] + zs[in_shell] * vzs[in_shell])/rs[in_shell]
        Pks[i] = (rho_ks[i]/nks[i]) * np.sum(v_rsi**2)
    
    dr0 = drs[0]
    n0 = np.sum(rhats<dr0)
    rho_0 = n0/((4/3) * np.pi * dr0**3)
    
    in_sphere = (rs < dr0)
    v_rsi = (xs[in_sphere] * vxs[in_sphere] + ys[in_sphere] * vys[in_sphere] + zs[in_sphere] * vzs[in_sphere])/rs[in_sphere]
    P0 = (rho_0/n0)*np.sum(v_rsi**2)
    
    return rho_ks, rho_0, Pks, P0 

rho_ks, rho_0, Pks, P0 = pressure_profile(samples)

fig3, ax3 = plt.subplots(1,1)
ax3.scatter(x = rks, y =Pks/P0, marker = 'x')
ax3.plot(model.rhat, model.pressure(model.psi)/model.pressure(Psi), color = 'r')
ax3.set_yscale('log')
ax3.set_xlabel('\\hat{r}')
ax3.set_ylabel('$logP/P_0$')
ax3.set_title(f'N = {len(rhats)}')





# r1, r2 = region_boundaries(initial_region, Psi, rK)
# rs, vs, rhats_add, vhats_add = new_samples(n_star_initial, M, rK, Psi, r1, r2)

# rhats_combined = np.concatenate((rhats, rhats_add))
# vhats_combined = np.concatenate((vhats, vhats_add))

# rks_combined, normalised_density_combined = rad_prob_density_profile(rhats_combined, vhats_combined)


        




    





















