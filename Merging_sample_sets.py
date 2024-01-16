#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:16:25 2024

@author: s1984454
"""

import numpy as np

set_1 = 'dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0.3_epsilon_0.1_N_20000.txt'
set_2 = 'dimensional_samples_samples_to_combine.txt'

new_name = 'dimensional_samples_King_M_500_rK_1.2_Psi_5_mu_0.3_epsilon_0.1_N_40000.txt'

set_1 = np.loadtxt('Data/'+ set_1)
set_2 = np.loadtxt('Data/'+ set_2)

new_set = np.concatenate((set_1, set_2), axis  = 0)

np.savetxt('Data/' + new_name, new_set)