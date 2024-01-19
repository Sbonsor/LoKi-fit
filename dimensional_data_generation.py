#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:47:39 2023

@author: s1984454
"""
from LoKi import LoKi
from LoKi_samp import LoKi_samp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

class dimensional_data_generation:
    
    def __init__(self, N, M, rK, Psi, mu, epsilon, **kwargs):
        
        self._set_kwargs(N, M, rK, Psi, mu, epsilon, **kwargs)
        
        self.generate_dimensionless_samples()
        
        self.scale_dimensions()
        
        if self.validate:
            self.validation_of_samples()
        
        self.mask_samples()
        
    def _set_kwargs(self, N, M, rK, Psi, mu, epsilon, **kwargs):
        
        if (Psi - 9*mu/(4*np.pi*epsilon) < 0):
            raise ValueError("Invalid model: a_0 must not be negative")
        
        ### Dimensional parameters
        self.M = M
        self.rK = rK
        
        ### Dimensionless parameters
        self.Psi = Psi
        self.mu = mu
        self.epsilon = epsilon
        
        ### Other parameters
        self.N = N
        self.G = 4.3009e-3
        self.save = True
        self.validate = True
        self.fname = None
        
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
                
    def generate_dimensionless_samples(self):
        
        self.model =  LoKi(self.mu, self.epsilon, self.Psi)
        
        self.dimensionless_samples = []
        
        while (len(self.dimensionless_samples) < self.N):
            
            current_sample = LoKi_samp(self.model, N = 1, plot = False, scale_nbody = False)
            
            xhat = current_sample.x[0]
            yhat = current_sample.y[0]
            zhat = current_sample.z[0]
            vxhat = current_sample.vx[0]
            vyhat = current_sample.vy[0]
            vzhat = current_sample.vz[0]

            v_t = np.sqrt(vxhat**2 + vyhat**2)
            vhat = np.sqrt(vxhat**2 + vyhat**2 + vzhat**2)
            rhat = np.sqrt(xhat**2 + yhat**2 + zhat**2)
            psi_interp = interp1d(self.model.rhat, self.model.psi, fill_value = (self.Psi, 0), bounds_error = False)

            Jhat = v_t * rhat
            Ehat = np.clip(0.5* vhat**2 - psi_interp(rhat), a_max = 0, a_min = None)

            def f(r, Ehat, Jhat, psi_interp):
                
                function = 2*(Ehat + psi_interp(r)) - (Jhat/r)**2
                
                return function

            r_grid = np.linspace(1e-4, self.model.rhat[-1], 10000)
            function = 2*(Ehat + psi_interp(r_grid)) - (Jhat/r_grid)**2
            upper_bracket = r_grid[np.where(function == max(function))[0][0]]

            r_p = root_scalar(f = f, args = (Ehat, Jhat, psi_interp), bracket = [1e-4, upper_bracket], method = 'bisect').root

            if (r_p >= self.epsilon):
                self.dimensionless_samples.append([xhat, yhat, zhat, vxhat, vyhat, vzhat])
                #print(len(self.dimensionless_samples))
        
        self.dimensionless_samples = np.array(self.dimensionless_samples)
        
        return 1
    
    def scale_dimensions(self):
        
        self.A_hat = self.M/(self.model.M_hat * self.model.density(self.Psi) * self.rK**3)
        self.sqrt_a = np.sqrt( 9/(4 * np.pi * self.G * self.A_hat * self.model.density(self.Psi) * self.rK**2) )
        
        
        self.dimensional_samples = np.zeros((self.N, 6))
        
        self.x  = self.dimensional_samples[:,0] = self.dimensionless_samples[:,0] * self.rK
        self.y  = self.dimensional_samples[:,1] = self.dimensionless_samples[:,1] * self.rK
        self.z  = self.dimensional_samples[:,2] = self.dimensionless_samples[:,2] * self.rK
        self.vx = self.dimensional_samples[:,3] = self.dimensionless_samples[:,3] / self.sqrt_a
        self.vy = self.dimensional_samples[:,4] = self.dimensionless_samples[:,4] / self.sqrt_a
        self.vz = self.dimensional_samples[:,5] = self.dimensionless_samples[:,5] / self.sqrt_a
        
        if self.save:
            
            if self.fname == None:
                np.savetxt(f'Data/dimensional_samples_King_M_{self.M}_rK_{self.rK}_Psi_{self.Psi}_mu_{self.mu}_epsilon_{self.epsilon}_N_{self.N}.txt', self.dimensional_samples)
            
            else:
                np.savetxt('Data/dimensional_samples_' + self.fname + '.txt', self.dimensional_samples)
                
            print('Dimensional samples saved')
        
        return 1
    
    
    def mask_samples(self):
        
        for i in range(int(self.N/3)): 
            
            self.dimensional_samples[3*i, 2] = None # Mask z coordinate for 1/3 of samples
            
            self.dimensional_samples[3*i + 1, 2] = None # Mask z, vx, vy coordinates for another 1/3 of the samples.
            self.dimensional_samples[3*i + 1, 3] = None
            self.dimensional_samples[3*i + 1, 4] = None
                
        if self.save:
            
            if self.fname == None:
                np.savetxt(f'Data/masked_dimensional_samples_King_M_{self.M}_rK_{self.rK}_Psi_{self.Psi}_mu_{self.mu}_epsilon_{self.epsilon}_N_{self.N}.txt.txt', self.dimensional_samples)
            
            else:
                np.savetxt('Data/masked_dimensional_samples_' + self.fname + '.txt', self.dimensional_samples)
                
            print('Masked samples saved')
            
        return 1
    
    def validation_of_samples(self):
        
        def density_normalisation(r,rho):
            integrand = r**2*rho
            return np.trapz(y = integrand, x = r)
        
        fig1,ax1 = plt.subplots(1,1)
        ax1.plot(self.model.rhat,self.model.rhat**2 * self.model.rho_hat/density_normalisation(self.model.rhat,self.model.rho_hat))
        ax1.hist(np.sqrt(self.dimensionless_samples[:,0]**2 + self.dimensionless_samples[:,1]**2 + self.dimensionless_samples[:,2]**2), density = True, bins = 100)
        ax1.set_xlabel('$\hat{r}$')
        ax1.set_ylabel('Radius probability')
        
        self.m = self.M/self.N
        K_i = 0.5 * self.m * (self.vx**2 + self.vy**2 + self.vz**2)
        self.K = np.sum(K_i)

        self.U = 0
        for i in range(self.N-1):
            
            x1 = self.x[i]
            y1 = self.y[i]
            z1 = self.z[i]

            x2_array = self.x[i+1:]
            y2_array = self.y[i+1:]
            z2_array = self.z[i+1:]
            
            separations = np.sqrt((x1 - x2_array)**2 + (y1 - y2_array)**2 + (z1 - z2_array)**2)
            contribution_to_U = -self.G * self.m**2 / separations
            
            self.U = self.U + np.sum(contribution_to_U)
            
        radii = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.M_BH = self.A_hat * self.model.density(self.Psi) * self.rK**3 * self.mu
        U_BH = np.sum(-self.G*self.m * self.M_BH/radii)
        
        self.U = self.U + U_BH
            
        self.virial_ratio = -self.K/self.U
        
        print(f'Q_vir = {self.virial_ratio}')
        
        r_min = self.epsilon * self.rK
        sample_r_min = min(radii)
        
        print(f'Theory r_min = {r_min}')
        print(f'sample r_min = {sample_r_min}')
        
        return 1
    
### Generate data    
N = 200000
M = 500
rK = 1.2 
Psi = 5
mu = 0.3
epsilon = 0.1
G = 4.3009e-3
#fname = 'samples_to_combine'

sampling = dimensional_data_generation(N, M, rK, Psi, mu, epsilon, save = True, validate = True)
