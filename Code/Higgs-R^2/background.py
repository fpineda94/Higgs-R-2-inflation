import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq, root_scalar
from Potential import Potential




class Background:

    def __init__(self, potential: Potential, lambda_h, xi, xi_s,  alpha = np.sqrt(2/3), N0 = 0, Nfin = 100):

        self.potential = potential
        self.lambda_h = lambda_h
        self.xi = xi
        self.xi_s = xi_s
        self.alpha = alpha
        self.N0 = N0
        self.Nfin = Nfin

        self.solution = None
        self._solver()
    
    def _eps_H(self, phi, dphidN, dhdN):        
        return 0.5*(dphidN**2 + np.exp(-self.alpha*phi)*dhdN**2)
    

    def _H(self, phi, dphidN, h, dhdN):
        V = self.potential.evaluate(phi, h)
        eps_H = self._eps_H(phi, dphidN, dhdN)
        return np.sqrt(V/(3 - eps_H))


    def _dotsigma(self, phi, dphidN, h, dhdN):
        H = self._H(phi, dphidN, h, dhdN)
        eps_H = self._eps_H(phi, dphidN, dhdN)

        return H*np.sqrt(2*eps_H)
    
    
    def _V_sigma(self, phi, dphidN, h, dhdN):
        H = self._H(phi, dphidN, h, dhdN)
        dotsigma = self._dotsigma(phi, dphidN, h, dhdN)
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h) 
        return H*(dphidN*dVdphi +dhdN*dVdh)/dotsigma
    
    def _eta_sigma(self, phi, dphidN, h, dhdN):
        V_sigma = self._V_sigma(phi, dphidN, h, dhdN)
        H = self._H(phi, dphidN, h, dhdN)
        dotsigma = self._dotsigma(phi, dphidN, h, dhdN)
        return (3*H*dotsigma + V_sigma)/(H*dotsigma)


    def _eta_perp(self, phi, dphidN, h, dhdN):
        
        H = self._H(phi, dphidN, h, dhdN)
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h) 

        return np.exp(0.5*self.alpha*phi)*(dVdh*dphidN - np.exp(-self.alpha*phi)*dVdphi*dhdN)/(H**2*(dphidN**2 + np.exp(-self.alpha*phi)*dhdN**2))

    
    def _m_s(self, phi, dphidN, h, dhdN):
        
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h)
        d2Vdphi2 = self.potential.second_derivative_phi(phi, h)
        d2Vdh2 = self.potential.second_derivative_h(phi, h)
        d2Vdphih = self.potential.second_derivative_phi_h(phi, h)

        H = self._H(phi, dphidN, h, dhdN)
        dotsigma = self._dotsigma(phi, dphidN, h, dhdN)
        return (H/dotsigma)**2*(dhdN**2*np.exp(-self.alpha*phi)*d2Vdphi2 + np.exp(self.alpha*phi)*dphidN**2*d2Vdh2 
                                - 2*dphidN*dhdN*d2Vdphih - 0.5*self.alpha*dVdphi*dphidN**2 - self.alpha*dhdN*dphidN*dVdh)  
    
    def _m_iso(self, phi, dphidN, h, dhdN):
        m_s = self._m_s(phi, dphidN, h, dhdN)
        H = self._H(phi, dphidN, h, dhdN)
        eta_perp = self._eta_perp(phi, dphidN, h, dhdN)
        eps_H = self._eps_H(phi, dphidN, dhdN)
        mu = - H**2*eps_H/3

        return m_s + mu - (H*eta_perp)**2




    def _EDOs(self, N, Y):

        [phi, dphidN, h, dhdN, H] = Y

        eps_H = self._eps_H(phi, dphidN, dhdN)
        dVdh = self.potential.derivative_h(phi, h)
        dVdphi = self.potential.derivative_phi(phi, h)
    
        #background
        d2phidN2 = -(3 - eps_H)*dphidN - dVdphi/H**2 - 0.5*self.alpha*np.exp(-self.alpha*phi)*dhdN
        d2hdN2 = -(3 - eps_H - self.alpha*dphidN)*dhdN - np.exp(self.alpha*h)*dVdh/H**2
        dHdN = -0.5*H*(dphidN**2 + np.exp(-self.alpha*phi)*dhdN**2)

        return [dphidN, d2phidN2, dhdN, d2hdN2, dHdN]


    def _InitialConditions(self):

        phi0 = 5.7
        dphi0dN, dh0dN = 0.0013, 0
        h0 = np.sqrt(self.xi*(np.exp(self.alpha*phi0) -1)/(self.xi**2 + 4*self.lambda_h*self.xi_s));   #Initial Higgs value along the vallyes of the potential 
        H0 = self._H(phi0, dphi0dN, h0, dh0dN)
        
        return [phi0, dphi0dN, h0, dh0dN ,H0]
    

    def _solver(self):

        Y0 = self._InitialConditions()
        N_span = [self.N0, self.Nfin]
        N_eval = np.linspace(self.N0, self.Nfin, 1000)

        self.solution = solve_ivp(self._EDOs, N_span,
            Y0,
            t_eval= N_eval,
            method= 'DOP853',
            rtol = 1e-6,
            atol = 1e-12,
            max_step = 0.1)
        
    @property
    def data(self):

        if self.solution is None:
            raise ValueError('Primero debes resolver el sistema')

        N = self.solution.t
        phi, dphidN, h, dhdN, H = self.solution.y
        a = np.exp(N)
        eps_H = self._eps_H(phi, dphidN, dhdN)
        dotsigma = self._dotsigma(phi, dphidN, h, dhdN)
        eta_perp = self._eta_perp(phi, dphidN, h, dhdN)
        eta_sigma = self._eta_sigma(phi, dphidN, h, dhdN)
        deta_perpdN = np.gradient(eta_perp, N)
        xi_perp = - deta_perpdN/(H*eta_perp)
        aH = a*H
        m_iso = self._m_iso(phi, dphidN, h, dhdN)

        return {'N': N, 'phi': phi, 'dphidN': dphidN, 'h': h, 'dhdN': dhdN, 'H': H, 'a': a, 'aH': aH, 'eps_H': eps_H, 'dotsigma': dotsigma, 'eta_perp': eta_perp, 'eta_sigma': eta_sigma, 'xi_perp': xi_perp, 'm_iso' : m_iso}
    
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
    
    def interpolation(self, x = 'Ne'):

        coords = {'N': self.data['N'], 'Ne': self.Ne}
        if x not in coords:
            raise ValueError('Interpolación debe ser respecto a N o Ne')
        
        x_vals = coords[x]
        variables = ['phi', 'dphidN', 'h', 'dhdN', 'H', 'a', 'aH', 'eps_H', 'dotsigma', 'eta_perp', 'eta_sigma', 'xi_perp', 'm_iso']

        return {
            var: interp1d(x_vals, self.data[var], kind = 'cubic', fill_value='extrapolate', bounds_error= False)
        for var in variables
        }