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


# import numpy as np
# import matplotlib.pyplot as plt
# import sympy as sp

# # Declarar las variables simbólicas
# phi, h = sp.symbols('phi h')
# alpha, lambda_h, M_p, xi_h, xi_s = sp.symbols('alpha lambda_h M_p xi_h xi_s')

# # Definir el potencial
# V = sp.exp(-2*alpha*phi)*(lambda_h*h**4 + M_p**4*(sp.exp(alpha*phi) - 1 - xi_h*h**2/M_p**2)**2/xi_s)/4

# # Derivadas
# dV_dphi = sp.diff(V, phi)
# dV_dh = sp.diff(V, h)

# d2V_dphi2 = sp.diff(dV_dphi, phi)
# d2V_dh2 = sp.diff(dV_dh, h)
# d2V_dphidh = sp.diff(V, phi, h, 2)
# d2V_dphidh_simplified = sp.simplify(d2V_dphidh)

# # Mostrar las expresiones de las derivadas
# print("Primera derivada con respecto a phi:")
# sp.pprint(dV_dphi)
# print("\nPrimera derivada con respecto a h:")
# sp.pprint(dV_dh)

# print("\nSegunda derivada con respecto a phi:")
# sp.pprint(d2V_dphi2)
# print("\nSegunda derivada con respecto a h:")
# sp.pprint(d2V_dh2)
# print("\nSegunda derivada cruzada:")
# sp.pprint(d2V_dphidh_simplified)


# # Convertir a funciones numéricas
# V_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), V, 'numpy')
# dV_dphi_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), dV_dphi, 'numpy')
# dV_dh_func = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), dV_dh, 'numpy')
# d2V_dphi2 = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dphi2, 'numpy')
# d2V_dh2 = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dh2, 'numpy')
# d2V_dphidh = sp.lambdify((phi, h, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dphidh, 'numpy')

import sympy as sp

# Declarar las variables simbólicas
phi, h = sp.symbols('phi h')
alpha, lambda_h, M_p, xi_h, xi_s = sp.symbols('alpha lambda_h M_p xi_h xi_s')

# Definir el potencial
V = sp.exp(-2*alpha*phi)*(lambda_h*h**4 + M_p**4*(sp.exp(alpha*phi) - 1 - xi_h*h**2/M_p**2)**2/xi_s)/4

# Derivadas de primer orden
dV_dphi = sp.diff(V, phi)
dV_dh   = sp.diff(V, h)

# Derivadas de segundo orden
d2V_dphi2 = sp.diff(dV_dphi, phi)
d2V_dh2   = sp.diff(dV_dh, h)
d2V_dphidh = sp.diff(V, phi, h)

# === 1. Resolver V_h = 0 para h(phi) ===
sol_h_list = sp.solve(sp.Eq(dV_dh, 0), h)

print("Soluciones h(phi):")
for i, sol in enumerate(sol_h_list):
    print(f"Rama {i+1}:")
    sp.pprint(sol)
    print()

# === 2. Elegir una rama para la trayectoria ===
h_phi_expr = sol_h_list[2]  # Cambiar índice si se quiere otra solución

# === 3. Sustituir h(phi) en las segundas derivadas ===
Vphiphi_on_traj = sp.simplify(d2V_dphi2.subs(h, h_phi_expr))
Vhh_on_traj     = sp.simplify(d2V_dh2.subs(h, h_phi_expr))
Vphih_on_traj   = sp.simplify(d2V_dphidh.subs(h, h_phi_expr))


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
sp.pprint(d2V_dphidh)



# === 4. Mostrar expresiones simbólicas ===
print("\nV_phiphi evaluado en h(phi):")
sp.pprint(Vphiphi_on_traj)

print("\nV_hh evaluado en h(phi):")
sp.pprint(Vhh_on_traj)

print("\nV_phih evaluado en h(phi):")
sp.pprint(Vphih_on_traj)

