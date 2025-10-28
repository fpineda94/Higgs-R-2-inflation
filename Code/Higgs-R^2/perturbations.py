import numpy as np 
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import brentq 
import matplotlib.pyplot as plt
import os

from background import Background
from Potential import Potential

class Perturbations:

    """
    Clase que calcula el espectro primordial de potencias para el modelo de dos campos Higgs-R^2
    """ 

    def __init__(self, potential : Potential, background: Background, N_CMB, k_CMB = 0.05, N_inside = 4):

        self.potential = potential
        self.background = background
        self.solution = None
        self._data_interpolated() 

        #Efolds configuration
        self.N_CMB = N_CMB
        self.N_inside = N_inside
        self.Nend = self.background.N_end
        self.Nhc = self.Nend - self.N_CMB

        #k modes configuration
        self.k_CMB = k_CMB
        self.k_pivot = self.aH(self.Nhc) 
        self.norma = self.k_CMB/self.k_pivot    #Normalization factor to convert k modes in Mpc^-1
        self.k_min, self.k_max = self.norma*self.aH(self.Nhc - 7), self.norma*self.aH(self.Nend - 4)
        self.k_modes = np.logspace(np.log10(self.k_min), np.log10(self.k_max), num = 1000)  #List modes in Mpc^-1


    def _data_interpolated(self, vars = None, x = 'N'):
        if vars is None:
            vars =  ['phi', 'dphidN', 'h', 'dhdN', 'H', 'a', 'aH', 'eps_H', 'dotsigma', 'eta_perp', 'eta_sigma', 'm_iso']
        bg_interp = self.background.interpolation(x)
        for i in vars:
            if i not in bg_interp:
                raise ValueError(f'The variable {i} is not available')
            setattr(self, i, bg_interp[i])
    
    @property
    def _ai(self):
        '''
        Next we need to fix the initial scale factor. 
        The initial scale factor is fixed by demanding a certain mode (pivot mode) leaves the Hubble scale at a particular time during the evolution.
        We will impose 0.05 $Mpc^{-1}$ mode leaves the Hubble radius 60 _efolds_ before the end of inflation.
        '''
        return self.k_CMB/(np.exp(self.Nhc)*self.H(self.Nhc))
    

    
    def _z(self, a, phi, dphidN, dhdN):
        alpha = np.sqrt(2/3)
        return a*np.sqrt(dphidN**2 + np.exp(-alpha*phi)*dhdN**2)
    


    def _ODEs(self, N, Y, k):
        
        [phi, dphidN, h, dhdN, Rk_re, Rk_re_N, Rk_im, Rk_im_N, Qk_re, Qk_re_N, Qk_im, Qk_im_N] = Y

        V = self.potential.evaluate(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h)
        dVdh = self.potential.derivative_h(phi, h)
        eps_H = self.eps_H(N)

        a = self._ai*np.exp(N)
        z = self._z(a, phi, dphidN, dhdN)
        

        pass


        


    