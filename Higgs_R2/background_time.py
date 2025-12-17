import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

from Higgs_R2.Potential import Potential
    


class Background:
    def __init__(self, potential: Potential,lambda_h, xi, xi_s, phi0, h0 = None, alpha=np.sqrt(2/3)):

        self.potential = potential
        self.phi0 = phi0

        self.lambda_h = lambda_h
        self.xi = xi
        self.xi_s = xi_s
        self.alpha = alpha

        if h0 is None:
            self.h0 = np.sqrt(self.xi*(np.exp(self.alpha*self.phi0) -1)/(self.xi**2 + 4*self.lambda_h*self.xi_s))
        else:
            self.h0 = h0

        self.solution = None
        self._solver()
    
    # Dimensionless Hubble parameter
    def _H(self, phi, y, h, z):
        V = self.potential.evaluate(phi, h)
        return np.sqrt((0.5 * y**2 + 0.5 * np.exp(-self.alpha * phi) * z**2 + V) / 3)


    def _dot_sigma(self, phi, y, z):
        return np.sqrt(y**2 + np.exp(-self.alpha*phi)*z**2 ) + 1e-12
    
    def _eps_H(self, phi, y, h, z):

        dot_sigma = self._dot_sigma(phi, y, z)
        H = self._H(phi, y, h, z)
        return 0.5*dot_sigma**2/H**2
    
    
    def _eta_perp(self, phi, y, h, z):

        H = self._H(phi, y, h, z)
        dot_sigma = self._dot_sigma(phi, y, z)
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h)

        return np.exp(0.5*self.alpha*phi)*(y*dVdh - np.exp(-self.alpha*phi)*z*dVdphi)/(H*dot_sigma**2)

    def _V_sigma(self, phi, y, h, z):
        dotsigma = self._dot_sigma(phi, y,  z)
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h) 
        return (y*dVdphi + z*dVdh)/dotsigma
    
    
    def _eta_sigma(self, phi, y, h, z):
        V_sigma = self._V_sigma(phi, y, h, z)
        H = self._H(phi, y, h, z)
        dotsigma = self._dot_sigma(phi, y, z)
        return (3*H*dotsigma + V_sigma)/(H*dotsigma)


    def _m_s(self, phi, y, h, z):
        
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h)
        d2Vdphi2 = self.potential.second_derivative_phi(phi, h)
        d2Vdh2 = self.potential.second_derivative_h(phi, h)
        d2Vdphih = self.potential.second_derivative_phi_h(phi, h)

        dot_sigma = self._dot_sigma(phi, y, z)

        return (np.exp(self.alpha*phi)*y**2*d2Vdh2 + np.exp(-self.alpha*phi)*z**2*d2Vdphi2 - 2*y*z*d2Vdphih - 0.5*self.alpha*(y**2*dVdphi + 2*y*z*dVdh))/(dot_sigma**2)
    
    def _m_iso(self, phi, y, h, z):
        
        m_s = self._m_s(phi, y, h, z)
        H = self._H(phi, y, h, z)
        eps_H = self._eps_H(phi, y, h, z)
        eta_perp = self._eps_H(phi, y, h, z)
        mu = - H**2*eps_H/3

        return m_s + mu - (H*eta_perp)**2
    

    #Define the system of ode's
    def _ODEs(self, t, Y):

        [phi, y, h, z, H, N] = Y

        dVdphi = self.potential.derivative_phi(phi, h)
        dVdh = self.potential.derivative_h(phi, h)

        dphidt = y 
        dhdt = z 
        dydt = -3*H*y - 0.5*self.alpha*np.exp(-self.alpha*phi)*z**2 - dVdphi;
        dzdt = -3*H*z + self.alpha*y*z - np.exp(self.alpha*phi)*dVdh;
        dHdt = -0.5*(y**2 + np.exp(-self.alpha*phi)*z**2);
        dNdt = H;

        return [dphidt, dydt, dhdt, dzdt, dHdt, dNdt]
    

    #Define a suitable initial conditions for the model
    def _InitialConditions(self, phi0, h0):

        V = self.potential.evaluate(phi0, h0)
        dVdphi = self.potential.derivative_phi(phi0, h0)
        dVdh = self.potential.derivative_h(phi0, h0)

        dphi0dN = 0 #-dVdphi/V
        dh0dN = 0 #- np.exp(self.alpha*phi0)*dVdh/V
       
        H0 = self._H(phi0, dphi0dN, h0, dh0dN)
        
        return [phi0, dphi0dN, h0, dh0dN, H0, 0]
    

    
    def _solver(self):

        t_span = [0, 2e7]
        teval = np.linspace(0, 2e7, 10000)
        Y0 = self._InitialConditions(self.phi0, self.h0)

        self.solution = solve_ivp(
            self._ODEs, t_span, 
            Y0,
            t_eval= teval, 
            method='DOP853', 
            rtol=1e-6, 
            atol=1e-12)
        
    @property
    def data(self):
        
        if self.solution is None:
            raise ValueError('Primero debes resolver el sistema de ecuaciones')

        t = self.solution.t
        phi, y, h, z, H, N = self.solution.y

        eps_H = self._eps_H(phi, y, h, z)
        dot_sigma = self._dot_sigma(phi, y, z)
        eta_perp = self._eta_perp(phi, y, h, z)
        m_iso = self._m_iso(phi, y, h, z)
        aH = np.exp(N)*H
        eta_sigma = self._eta_sigma(phi, y, h, z)

        return {'t': t, 'N': N, 'phi': phi, 'y': y, 'h': h, 'z' : z, 'H': H, 'a' : np.exp(N),  'aH': aH, 'eps_H' : eps_H, 'dot_sigma': dot_sigma, 'eta_perp': eta_perp, 'eta_sigma': eta_sigma, 'm_iso': m_iso}
    



    @property
    def N_end(self):
        if self.solution is None:
            raise ValueError('Primero debes resolver el background')
        
        N = self.data['N']
        eps_H = self.data['eps_H']
        idx = np.argmax(eps_H >= 1)
        Nend = N[idx]

        return Nend
    

    @property
    def Ne(self):

        Nend = self.N_end

        if Nend is None:
            raise ValueError('Inflation does not end in the given range')
        Ne = Nend - self.data['N']
        return Ne
    

    
    def interpolation(self, x = 'N'):

        coords = {'N': self.data['N'], 'Ne': self.Ne}
        if x not in coords:
            raise ValueError('Interpolación debe ser respecto a N o Ne')
        
        x_vals = coords[x]
        variables = ['phi', 'y', 'h', 'z', 'H', 'a', 'aH', 'eps_H', 'dot_sigma', 'eta_perp', 'eta_sigma', 'm_iso']

        return {
            var: interp1d(x_vals, self.data[var], kind = 'cubic', fill_value='extrapolate', bounds_error= False)
        for var in variables
        }