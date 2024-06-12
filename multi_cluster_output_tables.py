#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:01:19 2024

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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')
plt.rc('font', size=14)

def draw_r_v_line(ax, theta, color, lwidth = 0.2, label = None):
    
    M = theta[0]
    rK = theta[1]
    Psi = theta[2]
    model = LoKi(0, 1e-6, Psi, pot_only = True)
    Mhat = np.trapz(y = 4*np.pi*model.rhat**2 * model.density(model.psi) / model.density(model.Psi) , x = model.rhat)
    a = (9 * rK * Mhat)/(4*np.pi*G*M)
    
    r = np.linspace(1e-6, model.rhat[-1], 1000) * rK
    psi = np.interp(r/rK, model.rhat, model.psi)
    v = np.sqrt(2*psi)/np.sqrt(a)
        
    ax.plot(r, v, c = color, linewidth = lwidth, label = label)
    
    return r, v

def process_single_cluster(cluster, results_dict, idx):
    cluster_data = results_dict[cluster]
    
    theta_bar = cluster_data['mean']
    mean = theta_bar[idx]

    
    MAP = cluster_data['MAP']
    M_MAP = MAP[idx]

    covariance = cluster_data['covariance']
    var = covariance[idx,idx]
    
    CV = np.sqrt(var)/mean
    
    ### Read parameter file
    cluster_params = pd.read_csv (cluster_params_file, sep = '\t',comment='#')
    ### Select King models
    cluster_params = cluster_params[cluster_params['Mod'] == 'K ']
    ### Eliminate leading or trailing whitespace from strings
    cluster_params = cluster_params.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    true_params = cluster_params[cluster_params['Cluster'] == cluster]
    
    true_params = [ 10 ** true_params['Mtot'].astype(float).to_numpy()[0], true_params['r0pc'].astype(float).to_numpy()[0], true_params['W0/gamma'].astype(float).to_numpy()[0]]
    
    true = true_params[idx]
    
    error = abs((mean-true)/true)*100
    
    if idx == 0:
        mean = np.log10(mean)
        true = np.log10(true)
    
    output = [mean, true, error, np.sqrt(var), CV]
    
    return output

def output_table(data_file, cluster_params_file, idx):

    with open(data_file, 'rb') as handle:
         results_dict = pickle.load(handle)
         
    clusters = list(results_dict.keys())[1:]
    
    means = []
    trues = []
    errors = []
    stds = []
    CVs = []
    
    for cluster in clusters:
        
        cluster_output = process_single_cluster(cluster, results_dict, idx)
        
        means.append(cluster_output[0])
        trues.append(cluster_output[1]) 
        errors.append(cluster_output[2]) 
        stds.append(cluster_output[3])
        CVs.append(cluster_output[4])
    
    cluster_names = [x[3:] for x in clusters]    
    
    table_data = {'Cluster':cluster_names, r'$\bar{\theta}$':means, '$\theta^*$':trues, '$\theta$ \% error':errors, '$\sigma_{\theta}$':stds, r'$\sigma_{\theta}/\bar{\theta}$':CVs}
    
    table_df = pd.DataFrame(table_data)
    
    table_df.style.to_latex()
    
    latex_table = table_df.to_latex(
        index=False,  # To not include the DataFrame index as a column in the table
        caption="",  # The caption to appear above the table in the LaTeX document
        label="tab:multi_cluster_fits",  # A label used for referencing the table within the LaTeX document
        position="h",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
        column_format=" c c c c c c",  # The format of the columns: left-aligned with vertical lines between them
        escape=False,  # Disable escaping LaTeX special characters in the DataFrame
        float_format="{0:.4g}".format  # Formats floats to
    )
    
    print(latex_table)
    
    return 0



fname = 'multi_cluster_run.pkl'
cluster_params_file = 'Data/asu.tsv'
idx = 2 # 0=M, 1=rK, 2=Psi

### Physical constants.
G = 4.3009e-3

### Output numerical tables for all 10 clusters
output_table(fname, cluster_params_file, idx)


### Choose a cluster and output relevant figures.
cluster = 'MW-NGC104'
data_file = fname

with open(data_file, 'rb') as handle:
     results_dict = pickle.load(handle)

cluster_params = pd.read_csv (cluster_params_file, sep = '\t',comment='#')
     
cluster_data = results_dict[cluster]
npoints_axis = results_dict['npoints']
samples = cluster_data['samples']

rs = samples[0]
vs = samples[1]

### Select King models
cluster_params = cluster_params[cluster_params['Mod'] == 'K ']
### Eliminate leading or trailing whitespace from strings
cluster_params = cluster_params.applymap(lambda x: x.strip() if isinstance(x, str) else x)

true_params = cluster_params[cluster_params['Cluster'] == cluster]

M_max = 10**(true_params['Mtot'].astype(float).to_numpy()[0] + 2*true_params['E_Mtot'].astype(float).to_numpy()[0])
M_min = 10**(true_params['Mtot'].astype(float).to_numpy()[0] - 2*true_params['e_Mtot'].astype(float).to_numpy()[0])
rK_min = max(true_params['r0pc'].astype(float).to_numpy()[0] - 5*true_params['e_r0pc'].astype(float).to_numpy()[0], 0.01)
rK_max = true_params['r0pc'].astype(float).to_numpy()[0] + 5*true_params['E_r0pc'].astype(float).to_numpy()[0]
Psi_min = max(true_params['W0/gamma'].astype(float).to_numpy()[0] - 5*true_params['e_W0/gamma'].astype(float).to_numpy()[0], 0.1)
Psi_max = true_params['W0/gamma'].astype(float).to_numpy()[0] + 5*true_params['E_W0/gamma'].astype(float).to_numpy()[0]

M_axis = np.linspace(M_min, M_max, npoints_axis)
Psi_axis = np.linspace(Psi_min, Psi_max, npoints_axis)
rK_axis = np.linspace(rK_min, rK_max, npoints_axis)

X,Y,Z = np.meshgrid(M_axis, rK_axis, Psi_axis)

Ms = X.flatten()
RKs = Y.flatten()
PSIS = Z.flatten()


allowed_region = cluster_data['allowed_region']
log_ls = cluster_data['log_ls']
MAP = cluster_data['MAP']

theta_bar = cluster_data['mean']
theta_star = [10 ** true_params['Mtot'].astype(float).to_numpy()[0], true_params['r0pc'].astype(float).to_numpy()[0], true_params['W0/gamma'].astype(float).to_numpy()[0] ] 

##### Figure 1: r-v plane, true and fitted, with samples
fig, ax = plt.subplots(1,1)
ax.set_xlabel('$r$')
ax.set_ylabel('$v$')
draw_r_v_line(ax, theta_bar, 'k', lwidth = 0.2, label = '$\\bar{\\theta}$')
draw_r_v_line(ax, theta_star, 'r', lwidth = 0.2, label = '$\\theta^*$')
ax.scatter(rs, vs, marker = '.', color = 'k', s = 0.1)
ax.legend()

##### Figure 2: Likelihood 3d figure.
fig2 = plt.figure()
ax2 = plt.axes(projection="3d")
p = ax2.scatter3D(Ms[allowed_region], RKs[allowed_region], PSIS[allowed_region], c=log_ls[allowed_region], alpha=0.1, marker='.')
ax2.scatter3D(theta_star[0], theta_star[1], theta_star[2] , c = 'k', alpha=1, marker='x', label = '$\\theta^*$')
ax2.scatter3D(theta_bar[0], theta_bar[1] , theta_bar[2] , c = 'r', marker = 'x', alpha = 1, label = '$\\bar{\\theta}$')
ax2.scatter3D(MAP[0], MAP[1] , MAP[2] , c = 'b', marker = 'x', alpha = 1, label = '$\\theta_{MAP}$')
cbar = fig2.colorbar(p, label = 'log$P(\\theta|X)$')
cbar.solids.set(alpha=1)
ax2.set_xlabel('M')
ax2.set_ylabel('$r_K$')
ax2.set_zlabel('$\\Psi$')
ax2.legend()







