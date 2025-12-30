import numpy as np
import matplotlib.pyplot as plt


ruta_input = 'Analysis/power_spectrum_end_data.txt' 
data = np.loadtxt(ruta_input)
k = data[:, 0]
Pk = data[:, 1]


altura_actual = 7160.05
altura_objetivo = 5731.44
factor_renormalizacion = altura_objetivo / altura_actual

print(f"Factor de corrección: {factor_renormalizacion}")

Pk_final = Pk * factor_renormalizacion
ruta_final = 'Analysis/Pr_dense_input_NORMALIZADO.dat'
np.savetxt(ruta_final, np.column_stack([k, Pk_final]))
