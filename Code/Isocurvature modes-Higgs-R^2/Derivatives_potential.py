"""import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Declarar las variables simbólicas
phi, h = sp.symbols('phi h')
alpha, lambda_h, M_p, xi_h, xi_s = sp.symbols('alpha lambda_h M_p xi_h xi_s')

# Definir el potencial
V = sp.exp(-2*alpha*phi)*(lambda_h*h**4 + M_p**4*(sp.exp(alpha*phi) - 1 - xi_h*h**2/M_p**2)**2/xi_s)/4

# Derivadas
dV_dphi = sp.diff(V, phi)
dV_dh = sp.diff(V, h)

d2V_dphi2 = sp.diff(dV_dphi, phi)
d2V_dh2 = sp.diff(dV_dh, h)
d2V_dhdp = sp.diff(V, phi, h)
d2V_dphi2_simplified = sp.simplify(d2V_dphi2)
d2V_dh2_simplified = sp.simplify(d2V_dh2)



# Mostrar las expresiones de las derivadas
print("Primera derivada con respecto a phi:")
sp.pprint(dV_dphi)
print("\nPrimera derivada con respecto a h:")
sp.pprint(dV_dh)

print("\nSegunda derivada con respecto a phi:")
sp.pprint(d2V_dphi2_simplified)
print("\nSegunda derivada con respecto a h:")
sp.pprint(d2V_dh2_simplified)
print('\nSegunda derivada cruzada:')
sp.pprint(d2V_dhdp)

# Convertir a funciones numéricas
V_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), V, 'numpy')
dV_dphi_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), dV_dphi, 'numpy')
dV_dh_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), dV_dh, 'numpy')

"""


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Declarar las variables simbólicas
phi, h = sp.symbols('phi h')
alpha, lambda_h, M_p, xi_h, xi_s = sp.symbols('alpha lambda_h M_p xi_h xi_s')

# Definir el potencial
V = sp.exp(-2*alpha*phi)*(lambda_h*h**4 + M_p**4*(sp.exp(alpha*phi) - 1 - xi_h*h**2/M_p**2)**2/xi_s)/4

# Derivadas
dV_dphi = sp.diff(V, phi)
dV_dh = sp.diff(V, h)

d2V_dphi2 = sp.diff(dV_dphi, phi)
d2V_dh2 = sp.diff(dV_dh, h)
d2V_dphidh = sp.diff(V, phi, h, 2)
d2V_dphidh_simplified = sp.simplify(d2V_dphidh)

# Mostrar las expresiones de las derivadas
print("Primera derivada con respecto a phi:")
sp.pprint(dV_dphi)
print("\nPrimera derivada con respecto a h:")
sp.pprint(dV_dh)

print("\nSegunda derivada con respecto a phi:")
sp.pprint(d2V_dphi2)
print("\nSegunda derivada con respecto a h:")
sp.pprint(d2V_dh2)
print("\nSegunda derivada cruzada:")
sp.pprint(d2V_dphidh_simplified)


# Convertir a funciones numéricas
V_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), V, 'numpy')
dV_dphi_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), dV_dphi, 'numpy')
dV_dh_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), dV_dh, 'numpy')
d2V_dphi2 = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dphi2, 'numpy')
d2V_dh2 = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dh2, 'numpy')
d2V_dphidh = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dphidh, 'numpy')
