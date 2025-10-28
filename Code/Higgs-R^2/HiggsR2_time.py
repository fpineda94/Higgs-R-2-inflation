import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq, root_scalar

from Potential import Potential
    


class Background:
    def __init__(self, potential: Potential,lambda_h, xi, xi_s, trayectory = None, alpha=np.sqrt(2/3)):

        self.potential = potential

        self.lambda_h = lambda_h
        self.xi = xi
        self.xi_s = xi_s
        self.alpha = alpha

        self.trajectory = trayectory
        self.solution = None
        self.solver()
    
    # Dimensionless Hubble parameter
    def _H(self, phi, y, h, z):
        V = self.potential.evaluate(phi, h)
        return np.sqrt((0.5 * y**2 + 0.5 * np.exp(-self.alpha * phi) * z**2 + V) / 3)


    def _dot_sigma(self, phi, y, z):
        return np.sqrt(y**2 + np.exp(-self.alpha*phi)*z**2)
    
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
    def _InitialCondition(self):
        phi0 = 5.7

        if self.trajectory == 1:

            h0 = np.sqrt(self.xi*(np.exp(self.alpha*phi0) -1)/(self.xi**2 + 4*self.lambda_h*self.xi_s));   #Initial Higgs value along the vallyes of the potential  
        
        elif self.trajectory == 2:
            h0 = 1.5

        elif self.trajectory == 3:
            h0 = 0
        
        elif self.trajectory == 4:
            h0 = 1e-10

        y0, z0, N0 = 0, 0, 0
        H0 = self._H(phi0, y0, h0, z0)

        return [phi0, y0, h0, z0, H0, N0]
    

    
    def solver(self):

        t_span = [0, 2e7]
        teval = np.linspace(0, 2e7, 10000)

        Y0 = self._InitialCondition()
        self.solution = solve_ivp(
            self._ODEs, t_span, 
            Y0,
            t_eval= teval, 
            method='DOP853', 
            rtol=1e-5, 
            atol=1e-10)
        
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

        return {'t': t, 'N': N, 'phi': phi, 'y': y, 'h': h, 'z' : z, 'H': H, 'a' : np.exp(N),  'aH': aH, 'eps_H' : eps_H, 'dot_sigma': dot_sigma, 'eta_perp': eta_perp, 'm_iso': m_iso}

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