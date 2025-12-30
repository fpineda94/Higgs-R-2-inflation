import numpy as np 
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import brentq 
import matplotlib.pyplot as plt
import os

from Higgs_R2.background import Background
from Higgs_R2.Potential import Potential

class Perturbations:

    def __init__(self, potential : Potential, background: Background, N_CMB, k_CMB = None, N_inside = 4):

        self.potential = potential
        self.background = background
        self.solution = None
        self.solution1 = None

        self._data_interpolated() 
        
        self.alpha = np.sqrt(2/3)

        #Efolds configuration
        self.N_CMB = N_CMB
        self.N_inside = N_inside
        self.Nend = self.background.N_end
        self.Nhc = self.Nend - self.N_CMB

        #k modes configuration
        if k_CMB is None:
            self.k_CMB = 0.05
        else:
            self.k_CMB = k_CMB

        self.k_pivot = self.aH(self.Nhc) 
        self.norma = self.k_CMB/self.k_pivot    #Normalization factor to convert k modes in Mpc^-1
        self.k_min, self.k_max = self.norma*self.aH(self.Nhc - 10), self.norma*self.aH(self.Nend - 4)
        self.k_modes = np.logspace(np.log10(self.k_min), np.log10(self.k_max), num = 500)  #List modes in Mpc^-1


        self._data_interpolated() 

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
    
        return a*np.sqrt(dphidN**2 + np.exp(-self.alpha*phi)*dhdN**2)
    


    def _ODEs(self, N, Y, k):
        
        [phi, dphidN, h, dhdN, Rk_re, Rk_re_N, Rk_im, Rk_im_N, Qk_re, Qk_re_N, Qk_im, Qk_im_N, hk_re, hk_re_N, hk_im, hk_im_N] = Y

        dVdphi = self.potential.derivative_phi(phi, h)
        dVdh = self.potential.derivative_h(phi, h)
        eps_H = self.eps_H(N)
        H = self.H(N)


        #Background
        d2phidN2 = -(3 - eps_H)*dphidN - dVdphi/H**2 - 0.5*self.alpha*np.exp(-self.alpha*phi)*dhdN
        d2hdN2 = -(3 - eps_H - self.alpha*dphidN)*dhdN - np.exp(self.alpha*phi)*dVdh/H**2

        a = self._ai*np.exp(N)

          # --- z_N / z ---
        exp_term = np.exp(-self.alpha * phi)
        G = dphidN**2 + exp_term * dhdN**2
        G_N = 2.0 * dphidN * d2phidN2 + exp_term * (2.0 * dhdN * d2hdN2 - self.alpha * dphidN * dhdN**2)
        z_N_over_z = 1.0 + 0.5 * (G_N / G)


        #Adiabatic perturbations
        Rk_re_NN = -(1 - eps_H + 2*(z_N_over_z))*Rk_re_N - ((k/(a*H))**2)*Rk_re - 2*self.eta_perp(N)*(Qk_re_N + (3 - self.eta_sigma(N))*Qk_re)/(np.sqrt(2*eps_H))
        Rk_im_NN = -(1 - eps_H + 2*(z_N_over_z))*Rk_im_N - ((k/(a*H))**2)*Rk_im - 2*self.eta_perp(N)*(Qk_im_N + (3 - self.eta_sigma(N))*Qk_im)/(np.sqrt(2*eps_H))

        #Isocurvature perturbations
        Qk_re_NN = -(3 - eps_H)*Qk_re_N - ((k/(a*H))**2 + self.m_iso(N)/H**2)*Qk_re + 2*np.sqrt(2*eps_H)*self.eta_perp(N)*Rk_re_N
        Qk_im_NN = -(3 - eps_H)*Qk_im_N - ((k/(a*H))**2 + self.m_iso(N)/H**2)*Qk_im + 2*np.sqrt(2*eps_H)*self.eta_perp(N)*Rk_im_N

        #Tensor perturbations
        hk_re_NN = - (3-(dphidN**2)*0.5)*hk_re_N-((k/(a*H))**2)*hk_re
        hk_im_NN = - (3-(dphidN**2)*0.5)*hk_im_N-((k/(a*H))**2)*hk_im

        return [dphidN, d2phidN2, dhdN, d2hdN2, Rk_re_N, Rk_re_NN, Rk_im_N, Rk_im_NN, Qk_re_N, Qk_re_NN, Qk_im_N, Qk_im_NN, hk_re_N, hk_re_NN, hk_im_N, hk_im_NN]
    

    #e-folds configuration
    
    def N_hc(self, k=None, include_invalid=True):

        '''
        Find the efold N at which the k-mode crosses the horizon.
        Returns (N_hc, k) for each mode.
        '''

        def func_to_root(N_val, k_val):
            return k_val - self.norma*self.aH(N_val)

        if k is not None:
            try:
                N_val = brentq(lambda N: func_to_root(N, k), 0, self.Nend)
                return (N_val, k)
            except ValueError as e:
                print(f"Warning: Could not find horizon crossing for k={k} in [0, {self.Nend}]. Error: {e}")
                return (np.nan, k) if include_invalid else None
        else:
            results = []
            for k_val in self.k_modes:
                try:
                    N_val = brentq(lambda N: func_to_root(N, k_val), 0, self.Nend)
                    results.append((N_val, k_val))
                except ValueError as e:
                    print(f"Warning: Could not find horizon crossing for k={k_val} in [0, {self.Nend}]. Error: {e}")
                    if include_invalid:
                        results.append((np.nan, k_val))
            return results
            


    def N_ini(self, k=None):
        '''
        Find the efold N_ini for a given k mode, 4 efolds before horizon crossing.
        '''
        if k is not None:
            n_hc = self.N_hc(k)[0] 
            return n_hc - self.N_inside if not np.isnan(n_hc) else np.nan
        else:
            return [
                N_hc - self.N_inside if not np.isnan(N_hc) else np.nan
                for N_hc, _ in self.N_hc()
        ]


    def N_shs(self, k =None):

        if k is not None:
            n_hc = self.N_hc(k)[0]
            return n_hc + 10 if not np.isnan(n_hc) else np.nan
        else:
            return [N_hc + 10 if not np.isnan(N_hc) else np.nan
                    for N_hc, _ in self.N_hc()]

    

    def Initial_conditions(self, k):

        N0 = self.N_ini(k)
        a0 = self._ai*np.exp(N0)

        #Background initial conditions
        phi0 = self.phi(N0)
        dphi0dN = self.dphidN(N0)
        h0 = self.h(N0)
        dh0dN = self.dhdN(N0)
        H0 = self.H(N0)
        Y0 = [phi0, dphi0dN, h0, dh0dN, H0] 
        _, d2phi0dN2, _, d2h0dN2, _ = self.background._EDOs(N0, Y0)
        z0 = self._z(a0, phi0, dphi0dN, dh0dN)

        exp_term = np.exp(-self.alpha * phi0)
        G = dphi0dN**2 + exp_term * dh0dN**2
        G_N = 2.0 * dphi0dN * d2phi0dN2 + exp_term * (2.0 * dh0dN * d2h0dN2 - self.alpha * dphi0dN * dh0dN**2)
        z_N_over_z_0 = 1.0 + 0.5 * (G_N / G)

        #Bunch-Davies vacuum for R and Q_s perturbations
        Rk_re_ic = (1/(np.sqrt(2*k)))/z0
        Rk_im_ic = 0
        Rk_re_N_ic = -Rk_re_ic*z_N_over_z_0
        Rk_im_N_ic = - np.sqrt(k/2)/(a0*H0*z0)

        Qk_re_ic = (1/(np.sqrt(2*k)))/a0
        Qk_im_ic = 0
        Qk_re_N_ic = -Qk_re_ic
        Qk_im_N_ic = -np.sqrt(k/2)/(a0**2*H0)

        #Initial conditions for tensor perturbations
        hk_re_ic = (1/(np.sqrt(2*k)))/a0
        hk_im_ic = 0
        hk_re_N_ic = -hk_re_ic
        hk_im_N_ic = -np.sqrt(k/2)/(a0**2*H0)  

        return [phi0, dphi0dN, h0, dh0dN, Rk_re_ic, Rk_re_N_ic, Rk_im_ic, Rk_im_N_ic, Qk_re_ic, Qk_re_N_ic, Qk_im_ic, Qk_im_N_ic, hk_re_ic, hk_re_N_ic, hk_im_ic, hk_im_N_ic]
    

    def Initial_conditions1(self, k, mode_index=0): 
    
            N0 = self.N_ini(k)
            a0 = self._ai*np.exp(N0)

            #Background initial conditions 
            phi0 = self.phi(N0)
            dphi0dN = self.dphidN(N0)
            h0 = self.h(N0)
            dh0dN = self.dhdN(N0)
            H0 = self.H(N0)
            Y0 = [phi0, dphi0dN, h0, dh0dN, H0] 
            _, d2phi0dN2, _, d2h0dN2, _ = self.background._EDOs(N0, Y0)
            z0 = self._z(a0, phi0, dphi0dN, dh0dN)

            exp_term = np.exp(-self.alpha * phi0)
            G = dphi0dN**2 + exp_term * dh0dN**2
            G_N = 2.0 * dphi0dN * d2phi0dN2 + exp_term * (2.0 * dh0dN * d2h0dN2 - self.alpha * dphi0dN * dh0dN**2)
            z_N_over_z_0 = 1.0 + 0.5 * (G_N / G)

            
            val_R = (1/(np.sqrt(2*k)))/z0
            der_R = -val_R * z_N_over_z_0
            der_im_R = - np.sqrt(k/2)/(a0*H0*z0)

            val_Q = (1/(np.sqrt(2*k)))/a0
            der_Q = -val_Q
            der_im_Q = -np.sqrt(k/2)/(a0**2*H0)

            if mode_index == 0:
           
                Rk_re_ic, Rk_re_N_ic = val_R, der_R
                Rk_im_ic, Rk_im_N_ic = 0, der_im_R
                
                Qk_re_ic, Qk_re_N_ic = 0.0, 0.0
                Qk_im_ic, Qk_im_N_ic = 0.0, 0.0
                
            elif mode_index == 1:
    
                Rk_re_ic, Rk_re_N_ic = 0.0, 0.0
                Rk_im_ic, Rk_im_N_ic = 0.0, 0.0
                
                Qk_re_ic, Qk_re_N_ic = val_Q, der_Q
                Qk_im_ic, Qk_im_N_ic = 0, der_im_Q
            else:
                raise ValueError("mode_index debe ser 0 o 1")

            hk_re_ic = (1/(np.sqrt(2*k)))/a0
            hk_im_ic = 0
            hk_re_N_ic = -hk_re_ic
            hk_im_N_ic = -np.sqrt(k/2)/(a0**2*H0)  

            return [phi0, dphi0dN, h0, dh0dN, Rk_re_ic, Rk_re_N_ic, Rk_im_ic, Rk_im_N_ic, Qk_re_ic, Qk_re_N_ic, Qk_im_ic, Qk_im_N_ic, hk_re_ic, hk_re_N_ic, hk_im_ic, hk_im_N_ic]
    

    
    def solver(self):

        k = self.k_CMB
        Y0 = self.Initial_conditions(k)
        N_ini = self.N_ini(k)
        N_span = [N_ini, self.Nend]
        N_eval = np.linspace(N_ini, self.Nend, 1000)

        self.solution = solve_ivp(lambda N, Y: self._ODEs(N, Y, k),  
                        N_span, 
                        Y0, 
                        t_eval= N_eval, 
                        method ='Radau',
                        rtol = 1e-8, 
                        atol = 1e-12, 
                        dense_output= True)   
        return self.solution
    


    def solver1(self, mode_index=0): 

        k = self.k_CMB
        Y0 = self.Initial_conditions1(k, mode_index=mode_index) 
        N_ini = self.N_ini(k)
        N_span = [N_ini, self.Nend] 
    
        N_eval = np.linspace(N_ini, N_span[1], 1000)

        self.solution1 = solve_ivp(lambda N, Y: self._ODEs(N, Y, k),  
                        N_span, 
                        Y0, 
                        t_eval= N_eval, 
                        method ='Radau',
                        rtol = 1e-8, 
                        atol = 1e-12, 
                        dense_output= True)   
        return self.solution1
    


    def compute_power_spectra1(self, k=None):
        if k is None: k = self.k_CMB

        sol1 = self.solver1(mode_index=0)

        Y_end1 = sol1.y[:, -1]
        R1 = Y_end1[4] + 1j*Y_end1[6]
        Q1 = Y_end1[8] + 1j*Y_end1[10]

        h1 = Y_end1[12] + 1j*Y_end1[14]

        sol2 = self.solver1(mode_index=1)
        Y_end2 = sol2.y[:, -1]
        R2 = Y_end2[4] + 1j*Y_end2[6]
        Q2 = Y_end2[8] + 1j*Y_end2[10]
        
        N_end = sol1.t[-1]
        H_end = self.H(N_end)
        dot_sigma_end = self.dotsigma(N_end)


        if abs(dot_sigma_end) < 1e-20:
            S1, S2 = 0j, 0j
        else:
            pref = H_end / dot_sigma_end
            S1 = pref * Q1
            S2 = pref * Q2

        prefactor = k**3 / (2 * np.pi**2)
        
        P_R = prefactor * (np.abs(R1)**2 + np.abs(R2)**2)        
        P_S = prefactor * (np.abs(S1)**2 + np.abs(S2)**2)
        
        Cross = (R1 * np.conj(S1)) + (R2 * np.conj(S2))
        P_RS = prefactor * np.real(Cross)

        P_t = 8 * prefactor * (np.abs(h1)**2) 

        if (P_R + P_S) > 0:
            beta_iso = P_S / (P_R + P_S)
        else:
            beta_iso = 0

        if P_R * P_S > 0:
            cosDelta = P_RS / np.sqrt(P_R * P_S)
        else:
            cosDelta = 0.0
            
        r = P_t / P_R

        return {
            'P_R': P_R,
            'P_S': P_S,
            'P_RS': P_RS,
            'beta_iso': beta_iso,
            'cosDelta': cosDelta,
            'r': r,
            'debug_R_from_iso': np.abs(R2) 
        }
    

    def get_evolution_history(self, k=None):
            """
            Calcula la evolución temporal de P_R, P_S, beta y cosDelta
            sumando estadísticamente los dos modos independientes.
            """
            if k is None: k = self.k_CMB
            
            # 1. Definir malla temporal común (desde inicio hasta fin de inflación)
            N_ini = self.N_ini(k)
            N_eval = np.linspace(N_ini, self.Nend, 1000)
            
            # 2. DOBLE PASADA DEL SOLVER
            # Pasada 1: Modo Adiabático
            sol1 = self.solver1(mode_index=0)
            Y1 = sol1.sol(N_eval) # Evaluamos en la malla común
            
            # Pasada 2: Modo Isocurvatura
            sol2 = self.solver1(mode_index=1)
            Y2 = sol2.sol(N_eval)
            
            # 3. Reconstruir variables complejas
            # Índices basados en tu código: R(4,6), Q(8,10)
            R1 = Y1[4] + 1j*Y1[6]
            Q1 = Y1[8] + 1j*Y1[10]
            
            R2 = Y2[4] + 1j*Y2[6]
            Q2 = Y2[8] + 1j*Y2[10]
            
            # 4. Background y S (Isocurvatura física)
            # Vectorizamos para evaluar H y dotsigma en todo el array N_eval
            H_arr = self.H(N_eval)
            dsigma_arr = self.dotsigma(N_eval)

            # Evitar división por cero si dotsigma cruza por 0
            epsilon = 1e-30
            dsigma_safe = np.where(np.abs(dsigma_arr) < epsilon, epsilon, dsigma_arr)
            
            pref_S = H_arr / dsigma_safe
            S1 = pref_S * Q1
            S2 = pref_S * Q2
            
            # 5. CÁLCULO DE ESPECTROS (SUMA ESTADÍSTICA)
            prefactor = k**3 / (2 * np.pi**2)
            
            # Suma de cuadrados (Amplitudes)
            P_R = prefactor * (np.abs(R1)**2 + np.abs(R2)**2)
            P_S = prefactor * (np.abs(S1)**2 + np.abs(S2)**2)
            
            # Suma de productos cruzados (Correlación)
            # Cross = R1*S1* + R2*S2*
            Cross = (R1 * np.conj(S1)) + (R2 * np.conj(S2))
            P_RS = prefactor * np.real(Cross)
            
            # 6. OBSERVABLES DERIVADOS
            
            # Beta isocurvatura
            total_power = P_R + P_S
            beta_iso = np.zeros_like(P_R)
            mask_power = total_power > 1e-60
            beta_iso[mask_power] = P_S[mask_power] / total_power[mask_power]
            
            # Cos Delta (Con filtro de seguridad para P_S -> 0)
            cosDelta = np.zeros_like(P_R)
            # Solo calculamos correlación si P_S y P_R son numéricamente significativos
            mask_corr = (P_R > 1e-60) & (P_S > 1e-60)
            
            cosDelta[mask_corr] = P_RS[mask_corr] / np.sqrt(P_R[mask_corr] * P_S[mask_corr])
            cosDelta = np.clip(cosDelta, -1.0, 1.0) # Limpieza numérica
            
            return {
                'N': N_eval,
                'P_R': P_R,
                'P_S': P_S,
                'beta_iso': beta_iso,
                'cosDelta': cosDelta
            }


    @property
    def data(self):
       
        '''
        Extract the data of the commuting curvature perturbation and its derivative as a function of the number of e-folds N
        and store them in a dictionary.
        '''

        if self.solution is None:
            raise ValueError('First you have to solve the system with solver method')
        k = self.k_CMB
        N = self.solution.t
        R_re = self.solution.y[4]
        dRdN_re = self.solution.y[5] 
        R_im = self.solution.y[6]
        dRdN_im = self.solution.y[7]
        Q_re = self.solution.y[8]
        dQdN_re = self.solution.y[9]
        Q_im = self.solution.y[10]
        dQdN_im = self.solution.y[11]
        h_re = self.solution.y[12]
        dhdN_re = self.solution.y[13]
        h_im = self.solution.y[14]
        dhdN_im = self.solution.y[15]

        H = self.H(N)
        dot_sigma = self.dotsigma(N)
        
        Rk = R_re + 1j*R_im
        Qk = Q_re + 1j*Q_im
        hk = h_re + 1j*h_im
        Sk = (H/dot_sigma)*Qk


        #Power spectrum
        P_R = k**3*np.abs(Rk)**2/(2*np.pi**2)
        P_S= k**3*np.abs(Sk)**2/(2*np.pi**2)
        P_t = 8*k**3*np.abs(hk)**2/(2*np.pi**2)

        #Primordial power spectrum and tensor to scalar ratio at the end of inflation
        Y_end = self.solution.sol(self.Nend)

        Rk_re_end = Y_end[4]
        Rk_im_end = Y_end[6]
        Qk_re_end = Y_end[8]
        Qk_im_end = Y_end[10]
        h_re_end = Y_end[12]
        h_im_end = Y_end[14]

        Rk_end = Rk_re_end + 1j*Rk_im_end
        Qk_end = Qk_re_end + 1j*Qk_im_end
        Sk_end = (H/dot_sigma)*Qk_end


        P_R_end = k**3*np.abs(Rk_end)**2/(2*np.pi**2)
        P_S_end = k**3*np.abs(Sk_end)**2/(2*np.pi**2)
        P_t_end = 8*k**3*(h_re_end**2 + h_im_end**2)/(2*np.pi**2)
        r_end = P_t_end/P_R_end

        return {'N': N, 'R_re' : R_re, 'dRdN_re': dRdN_re ,'R_im': R_im, 'dRdN_im': dRdN_im, 
                'Q_re' : Q_re, 'dQdN_re' : dQdN_re, 'Q_im' : Q_im, 'dQdN_im' : dQdN_im, 'h_re' : h_re, 'dhdN_re' : dhdN_re, 'h_im' : h_im, 'dhdN_im' : dhdN_im,'P_R': P_R, 
                    'P_S': P_S, 'P_t': P_t,  'P_R_end': P_R_end, 'P_S_end': P_S_end, 'P_t_end': P_t_end, 'r_end': r_end}
    
              

    def _Compute_Power_spectrum_at_end(self, k):

        Y0 = self.Initial_conditions(k)
        N_ini = self.N_ini(k)

        # For odeint we need the time as the first argument in the ODE        
        def ode_func(Y, N, k):
            return self._ODEs(N, Y, k)
        #We use an adaptative tolerance for the very small modes (k >> aH)
        tol = 1e-16/k

        # Solve the system with odeint (LSODA optimised in FORTRAN)
        sol = odeint(
            ode_func,
            Y0,
            np.linspace(N_ini, self.Nend, 1000),  
            args=(k,),
            atol=tol,
            mxstep= 10000000
            )   
        
        Y_end = sol[-1]
        Rk_re, Rk_im, Qk_re, Qk_im, hk_re, hk_im = Y_end[4], Y_end[6], Y_end[8], Y_end[10], Y_end[12], Y_end[14]
        
        sigma_end = self.dotsigma(self.Nend)
        H_end = self.H(self.Nend)

        R_k = Rk_re + 1j * Rk_im
        Q_k    = Qk_re + 1j * Qk_im

        S_k = (H_end/sigma_end)*Q_k

        P_R  = (k**3 / (2 * np.pi**2)) * np.abs(R_k)**2
        P_S  = (k**3 / (2 * np.pi**2)) * np.abs(S_k)**2
        P_RS = (k**3 / (2 * np.pi**2)) * np.real(S_k * np.conj(R_k)) 
        P_T = 8 * k**3 * (hk_re**2 + hk_im**2) / (2 * np.pi**2)


        return P_R, P_S, P_RS, P_T



 
    def Power_spectrum_end(self, save=False, filename=None):

        Pz = np.zeros_like(self.k_modes)
        Ps = np.zeros_like(self.k_modes)
        P_zs = np.zeros_like(self.k_modes)
        PT = np.zeros_like(self.k_modes)


        for i, k in enumerate(self.k_modes):
            Pz[i], Ps[i], P_zs[i], PT[i] = self._Compute_Power_spectrum_at_end(k)

        self._P_R_array = Pz
        self._P_S_array = Ps
        self._P_RS_array = P_zs
        self._P_t_array = PT


        if save:
            if filename is None:
                filename = 'power_spectrum_end_data.txt'

            filepath = os.path.join('Data', filename)
            np.savetxt(filepath,
                    np.column_stack([self.k_modes, Pz, Ps, P_zs]),
                    header='k_modes P_R(k) P_S(k) P_RS(k) ',
                    fmt='%.16e')

        return Pz, Ps, P_zs, PT
    


    @property
    def Spectral_tilts(self):
        
        '''
        Calculates the spectral indices n_s and n_t evaluated on the pivot scale k_pivot,
        using the spectrum already calculated with Power_spectrum().
        '''
        
        from scipy.interpolate import interp1d

        if not hasattr(self, '_P_R_array') or not hasattr(self, '_P_t_array'):
            raise RuntimeError("First you must run the Power_spectrum method to calculate the spectra.")

        k = self.k_modes
        P_s = self._P_R_array
        P_t = self._P_t_array
        k_pivot = self.k_CMB

        log_k = np.log(k)
        dlogPs = np.gradient(np.log(P_s), log_k)
        dlogPt = np.gradient(np.log(P_t), log_k)

        # Interpolation
        n_s_interp = interp1d(k, 1 + dlogPs, kind='cubic', bounds_error = False, fill_value="extrapolate")
        n_t_interp = interp1d(k, dlogPt, kind='cubic', bounds_error = False, fill_value="extrapolate")

        n_s_pivot = float(n_s_interp(k_pivot))
        n_t_pivot = float(n_t_interp(k_pivot))

        return {'n_s': n_s_pivot, 'n_t': n_t_pivot}
    
