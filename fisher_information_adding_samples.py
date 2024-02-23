#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:26:58 2024

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

def r_v(data):
    
    rs = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    vs = np.sqrt(data[:,3]**2 + data[:,4]**2 + data[:,5]**2)
    
    return rs, vs

def radii_separations(mu, Psi, rK):
    
    model = LoKi(mu, 1e-6, Psi)
    r1 = np.interp(0.5*model.M_hat, model.M_r,model.rhat) * rK
    r2 = np.interp(0.8*model.M_hat, model.M_r,model.rhat) * rK
    
    return r1, r2

def split_data_by_radius(data, r1, r2):
    
    rs, vs = r_v(data)
    
    inner_indices = np.where(rs<=r1)[0]
    intermediate_indices = np.where(np.logical_and(rs>r1, rs<=r2))[0]
    outer_indices = np.where(rs>r2)[0]
    
    return data[inner_indices,:], data[intermediate_indices,:], data[outer_indices,:]

def extract_random_subsample(full_sample, N_sample):
    
    sub_sample_indices = np.random.choice(len(full_sample), N_sample, replace = False)
    sub_sample = full_sample[sub_sample_indices,:]
    
    return sub_sample

def combine_information_matrix(data):
    
    Js = data[:,6:]

    matrices = []
    for i in range(len(Js)):    
        matrix = np.reshape(Js[i,:], (3,3))
        matrices.append(matrix)
        
    combined_matrix = sum(matrices)/len(data)
    det = np.linalg.det(combined_matrix)
        
    return combined_matrix, det

def run_addition_of_subsets(n_trials, full_sample_set, N_add, base_data):
    
    determinants_after_addition = []
    
    for i in range(n_trials):
        
        samples_to_add = extract_random_subsample(full_sample_set, N_add)
        combined_sample_set = np.concatenate((base_data, samples_to_add), axis  = 0)
        
        matrix, det = combine_information_matrix(combined_sample_set)
        determinants_after_addition.append(det)
        print(i)
        
    return np.array(determinants_after_addition)

nprocessors = 10
M = 500
rK = 1.2
Psi = 5
mu = 0
epsilon = 1e-6
theta = np.array([M, rK, Psi, mu, epsilon])
h = 1e-4
n_trials = 100
N_add = 10000

base_file = 'fisher_evaluations_Base_sample_set'
addition_file = f'fisher_evaluations_King_M_{M}_rK_{rK}_Psi_{Psi}_mu_{mu}_epsilon_{epsilon}_N_1000000_processor'

base_data = combine_data(base_file, nprocessors)
addition_data = combine_data(addition_file, nprocessors)

half_mass_radius, outer_region_radius = radii_separations(mu, Psi, rK)

inner_stars, intermediate_stars, outer_stars = split_data_by_radius(addition_data, rK, outer_region_radius)

base_information_matrix, base_information_matrix_det = combine_information_matrix(base_data)

adding_inner = run_addition_of_subsets(n_trials, inner_stars, N_add, base_data)
adding_intermediate = run_addition_of_subsets(n_trials, intermediate_stars, N_add, base_data)
adding_outer = run_addition_of_subsets(n_trials, outer_stars, N_add, base_data)

information_gain_inner = adding_inner - base_information_matrix_det
information_gain_intermediate = adding_intermediate - base_information_matrix_det
information_gain_outer = adding_outer - base_information_matrix_det

    








