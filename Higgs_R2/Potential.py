import numpy as np
import sympy as sp
from abc import ABC, abstractmethod




class Potential(ABC):
    @abstractmethod
    def evaluate(self, phi, h):
        pass

    @abstractmethod
    def derivative_phi(self, phi, h):
        pass 

    @abstractmethod
    def derivative_h(self, phi, h):
        pass 

    @abstractmethod
    def second_derivative_phi(self, phi, h):
        pass 

    @abstractmethod
    def second_derivative_h(self, phi, h):
        pass

    @abstractmethod
    def second_derivative_phi_h(self, phi, h):
        pass 

    

class Potential_function(Potential):
    def __init__(self, potential_func, dVdh, dVdphi, d2Vdphi2, d2Vdh2, d2Vdphidh,
                 V_expr=None, dVdphi_expr=None, dVdh_expr=None,
                 d2Vdphi2_expr=None, d2Vdh2_expr=None, d2Vdphidh_expr=None):
        
        self._potential_func = potential_func
        self._dVdphi = dVdphi
        self._dVdh = dVdh
        self._d2Vdphi2 = d2Vdphi2
        self._d2Vdh2 = d2Vdh2
        self._d2Vdphidh = d2Vdphidh

        # Guardar las expresiones simbólicas
        self.V_expr = V_expr
        self.dVdphi_expr = dVdphi_expr
        self.dVdh_expr = dVdh_expr
        self.d2Vdphi2_expr = d2Vdphi2_expr
        self.d2Vdh2_expr = d2Vdh2_expr
        self.d2Vdphidh_expr = d2Vdphidh_expr

    # Métodos numéricos
    def evaluate(self, phi, h):
        return self._potential_func(phi, h)

    def derivative_phi(self, phi, h):
        return self._dVdphi(phi, h)

    def derivative_h(self, phi, h):
        return self._dVdh(phi, h)

    def second_derivative_phi(self, phi, h):
        return self._d2Vdphi2(phi, h)

    def second_derivative_h(self, phi, h):
        return self._d2Vdh2(phi, h)

    def second_derivative_phi_h(self, phi, h):
        return self._d2Vdphidh(phi, h)

    # Métodos simbólicos (útiles para verificar)
    def get_symbolic_expression(self):
        return self.V_expr

    def get_symbolic_derivative_phi(self):
        return self.dVdphi_expr

    def get_symbolic_derivative_h(self):
        return self.dVdh_expr

    def get_symbolic_second_derivative_phi(self):
        return self.d2Vdphi2_expr

    def get_symbolic_second_derivative_h(self):
        return self.d2Vdh2_expr

    def get_symbolic_second_derivative_phi_h(self):
        return self.d2Vdphidh_expr

    @staticmethod
    def from_string(expr, param_values={}):
        try:
            phi, h = sp.symbols('phi h')
            param_symbols = {name: sp.symbols(name) for name in param_values.keys()}
            V_expr = sp.sympify(expr).subs(param_symbols)

            # Derivadas simbólicas
            dVdphi_expr = sp.diff(V_expr, phi)
            dVdh_expr = sp.diff(V_expr, h)
            d2Vdphi2_expr = sp.diff(dVdphi_expr, phi)
            d2Vdh2_expr = sp.diff(dVdh_expr, h)
            d2Vdphidh_expr = sp.diff(dVdphi_expr, h)

            # Funciones evaluables numéricamente
            args = (phi, h, *param_symbols.values())
            V_func = sp.lambdify(args, V_expr, 'numpy')
            dVdphi_func = sp.lambdify(args, dVdphi_expr, 'numpy')
            dVdh_func = sp.lambdify(args, dVdh_expr, 'numpy')
            d2Vdphi2_func = sp.lambdify(args, d2Vdphi2_expr, 'numpy')
            d2Vdh2_func = sp.lambdify(args, d2Vdh2_expr, 'numpy')
            d2Vdphidh_func = sp.lambdify(args, d2Vdphidh_expr, 'numpy')

            # Funciones con parámetros fijados
            V = lambda phi, h: V_func(phi, h, *param_values.values())
            dVdphi = lambda phi, h: dVdphi_func(phi, h, *param_values.values())
            dVdh = lambda phi, h: dVdh_func(phi, h, *param_values.values())
            d2Vdphi2 = lambda phi, h: d2Vdphi2_func(phi, h, *param_values.values())
            d2Vdh2 = lambda phi, h: d2Vdh2_func(phi, h, *param_values.values())
            d2Vdphidh = lambda phi, h: d2Vdphidh_func(phi, h, *param_values.values())

        except Exception as e:
            raise ValueError(f"Error al procesar el potencial: {e}")

        return Potential_function(
            V, dVdh, dVdphi, d2Vdphi2, d2Vdh2, d2Vdphidh,
            V_expr, dVdphi_expr, dVdh_expr, d2Vdphi2_expr, d2Vdh2_expr, d2Vdphidh_expr
        )