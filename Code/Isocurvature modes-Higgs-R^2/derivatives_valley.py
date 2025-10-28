import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Declarar variables simbólicas
phi, h = sp.symbols('phi h')  # Ahora h también es simbólica
alpha, lambda_h, M_p, xi_h, xi_s = sp.symbols('alpha lambda_h M_p xi_h xi_s', real=True, positive=True)

# Definir el potencial
V = sp.exp(-2*alpha*phi) * (lambda_h*(h**4) + M_p**4*(sp.exp(alpha*phi) - 1 - xi_h*(h**2)/M_p**2)**2/xi_s) / 4

# Derivadas simbólicas
d2V_dphi2 = sp.diff(V, phi, 2)  # Segunda derivada pura con respecto a phi
d2V_dh2 = sp.diff(V, h, 2)      # Segunda derivada pura con respecto a h
d2V_dhdp = sp.diff(sp.diff(V, h), phi)  # Derivada mixta

# Expresión para h como función de phi
h_expr = sp.sqrt(xi_h * (sp.exp(alpha*phi) - 1) / (xi_h**2 + lambda_h*xi_s))

# Sustituir h por la expresión en las derivadas
d2V_dphi2_at_h = d2V_dphi2.subs(h, h_expr)
d2V_dh2_at_h = d2V_dh2.subs(h, h_expr)
d2V_dhdp_at_h = d2V_dhdp.subs(h, h_expr)

# Crear funciones numéricas con lambdify
d2V_dphi2_func = sp.lambdify((phi, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dphi2_at_h, 'numpy')
d2V_dh2_func = sp.lambdify((phi, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dh2_at_h, 'numpy')
d2V_dhdp_func = sp.lambdify((phi, alpha, lambda_h, M_p, xi_h, xi_s), d2V_dhdp_at_h, 'numpy')

# Definir valores de los parámetros
params = {
    'alpha': 0.1,
    'lambda_h': 1e-3,
    'M_p': 1.0,
    'xi_h': 0.1,
    'xi_s': 1.0
}

# Crear rango de phi
phi_vals = np.linspace(-1, 6, 100)

# Evaluar las derivadas
d2V_dphi2_vals = d2V_dphi2_func(phi_vals, **params)
d2V_dh2_vals = d2V_dh2_func(phi_vals, **params)
d2V_dhdp_vals = d2V_dhdp_func(phi_vals, **params)

# Graficar las derivadas
plt.figure(figsize=(12, 6))

plt.plot(phi_vals, d2V_dphi2_vals, label=r'$\frac{\partial^2 V}{\partial \phi^2}$', color='blue')
plt.plot(phi_vals, d2V_dh2_vals, label=r'$\frac{\partial^2 V}{\partial h^2}$', color='red')
plt.plot(phi_vals, d2V_dhdp_vals, label=r'$\frac{\partial^2 V}{\partial h \partial \phi}$', color='green')

plt.xlabel(r'$\phi$', fontsize=14)
plt.ylabel('Valor de las derivadas', fontsize=14)
plt.title('Derivadas de $V(\phi, h)$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()