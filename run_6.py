#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:08:22 2024

@author: s1984454
"""

import numpy as np
from class_implementation import fitting

M0 = 500
rK0 = 1.2
Psi0 = 5
mu0 = 0.3
eps0 = 0.1

rho0_max = 170
rho0_min = 1
rK_max = 2
rK_min = 0.5
Psi_max = 9
Psi_min = 1


true_params = np.array([M0, rK0, Psi0, mu0, eps0])
prior_args = np.array([rho0_max, rho0_min, rK_max, rK_min, Psi_max, Psi_min])
initial_offset = np.array([2, 0.2, 0, 2,  0])
data_path = '/home/s1984454/LoKi-fit/Data/'
data_file  = f'dimensional_samples_King_M_{M0}_rK_{rK0}_Psi_{Psi0}_mu_{mu0}_epsilon_{eps0}_N_40000'
nsamp = 100000
nsamp_tune = 10000
target_acceptance_rate = 0.2
acceptance_rate_tol = 0.02
save_samples = True

# for i in range(10):
#     output_file = f'run_6_{i}.txt'
#     fitting(true_params, prior_args, initial_offset, data_path, data_file, nsamp, nsamp_tune, target_acceptance_rate, acceptance_rate_tol, save_samples, output_file)