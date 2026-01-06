import numpy as np
import matplotlib.pyplot as plt


ruta_input = 'Analysis/power_spectrum_end_xi_0.1.txt' 
data = np.loadtxt(ruta_input)
k = data[:, 0]
Pk = data[:, 1]


current_height = 7160.05
target_height = 5731.44
renormalizacion = target_height / current_height

print(f"Factor de corrección: {renormalizacion}")

Pk_final = Pk * renormalizacion
ruta_final = 'Analysis/Pr_dense_input_NORMALIZADO.dat'
np.savetxt(ruta_final, np.column_stack([k, Pk_final]))
