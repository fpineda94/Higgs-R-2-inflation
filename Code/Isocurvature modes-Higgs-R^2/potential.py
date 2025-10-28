import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LogLocator

# Definir parámetros
M_p = 1.0
alpha = np.sqrt(2/3*M_p**2)
lambda_h = 0.13
xi_s = 4e8
xi_h = np.sqrt(2e9 - xi_s)*lambda_h

# Definir el potencial
def V(phi, h):
    return np.exp(-2 * alpha * phi) * (
        lambda_h * h**4 + M_p**4 * (np.exp(alpha * phi) - 1 - xi_h * h**2 / M_p**2) ** 2 / xi_s
    ) / 4

# Crear un grid para phi y h
phi_vals = np.linspace(-1, 6, 80)
h_vals = np.linspace(-0.1, 0.1, 80)
phi_grid, h_grid = np.meshgrid(phi_vals, h_vals)

# Evaluar el potencial
V_vals = V(phi_grid, h_grid) / M_p**4

# Configurar rango de graficado para V
V_min, V_max = 1 / 10**11, 2 / 10**10
V_vals_clipped = np.clip(V_vals, V_min, V_max)

# Crear el gráfico
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Superficie del gráfico
surf = ax.plot_surface(
    phi_grid, h_grid, V_vals_clipped,
    cmap='winter',  # Color equivalente a "Aquamarine"
    edgecolor='none',
    rstride=1, cstride=1,
)

# Configuración de ejes
ax.set_xlabel(r'$\phi$', labelpad=10, fontsize=15, fontfamily="Times New Roman")
ax.set_ylabel(r'$h$', labelpad=10, fontsize=15, fontfamily="Times New Roman")
ax.set_zlabel(r'$V(\phi, h)/M_p^4$', labelpad=15, fontsize=15, fontfamily="Times New Roman")
ax.set_title(r'Potencial $V(\phi, h)$', fontsize=15, fontfamily="Times New Roman")

# Rango del eje z
ax.set_zscale('log')
ax.zaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

# Colorbar
cbar = fig.colorbar(surf, pad=0.1, aspect=10, shrink=0.8)
cbar.set_label(r'$V(\phi, h)/M_p^4$', fontsize=12, fontfamily="Times New Roman")

# Mostrar el gráfico
plt.show()