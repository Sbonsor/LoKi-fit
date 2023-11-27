from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import simps
from scipy.special import gammainc, gamma
import time

def means_covariace_information_matrix(samples):

    n_parameters = np.shape(samples)[1]     
    
    means = np.zeros(n_parameters)
    cov_matrix = np.zeros((n_parameters,n_parameters))
    
    for i in range(n_parameters):
        
        means[i] = np.mean(samples[:,i])
        i_centered = samples[:,i] - np.mean(samples[:,i])
        
        for j in range(n_parameters):
            
            j_centered = samples[:,j] - np.mean(samples[:,j])
            
            cov_matrix[i][j] = np.mean(i_centered*j_centered)
    
    information_matrix = np.linalg.inv(cov_matrix)
    
    return means, cov_matrix, information_matrix

samples = np.loadtxt('Data/SAMPLES_parallel_dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0.3_epsilon_0.1_N_20000_with_incomplete_data_and_mu.txt')

burn_in = 50000
stack_samp = np.stack(samples[burn_in:,:])

fig = corner.corner(stack_samp, labels = ['$M$','$r_K$','$\\Psi$', '$\\mu/\\epsilon$'], quantiles = [0.5])
# plt.savefig('Data/FIG_parallel_' + fname)
# fig = corner.corner(stack_samp, labels = ['$M$','$r_K$','$\\Psi$', '$\\mu$', '$\\epsilon$'], quantiles = [0.5])
# plt.savefig('Data/FIG_parallel_' + fname + '_with_incomplete_data.png')

# M_samples = samples[:,0]
# rc_samples = samples[:,1]
# Psi_samples = samples[:,2]

# means, cov_matrix, info_matrix = means_covariace_information_matrix(samples)

# M_mean = means[0]
# rc_mean = means[1]
# Psi_mean = means[2]

# metric = np.linalg.det(info_matrix)

        
