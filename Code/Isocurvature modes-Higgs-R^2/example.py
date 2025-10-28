import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir el potencial como una función
def potential(phi, xi):
    # Sustituir por la expresión real del potencial
    return np.exp(-phi**2 - xi**2)  # Ejemplo

# Crear una malla de valores
phi = np.linspace(-0.1, 0.1, 100)
xi = np.linspace(0, 6, 100)
phi_mesh, xi_mesh = np.meshgrid(phi, xi)
V = potential(phi_mesh, xi_mesh)

# Crear la figura y los ejes 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie del potencial
ax.plot_surface(phi_mesh, xi_mesh, V, cmap='copper', alpha=0.8)

# Agregar trayectorias (ejemplo)
# Sustituir con las soluciones reales
trajectory1_phi = np.linspace(-0.05, 0.05, 50)
trajectory1_xi = np.linspace(1, 5, 50)
trajectory1_V = potential(trajectory1_phi, trajectory1_xi)

ax.plot(trajectory1_phi, trajectory1_xi, trajectory1_V, color='red', lw=2, label='Trayectoria 1')

# Personalización del gráfico
ax.set_xlabel(r'$\varphi / M_p$')
ax.set_ylabel(r'$\hat{\xi} / M_p$')
ax.set_zlabel(r'$V$')
ax.view_init(elev=30, azim=45)
plt.legend()
plt.show()