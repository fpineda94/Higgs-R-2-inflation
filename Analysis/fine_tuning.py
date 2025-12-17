import numpy as np
import matplotlib.pyplot as plt


ruta_input = 'Analysis/Pr_dense_input_example2.dat' 
data = np.loadtxt(ruta_input)
k = data[:, 0]
Pk = data[:, 1]


altura_actual = 38388.80
altura_objetivo = 5731.44
factor_renormalizacion = altura_objetivo / altura_actual

print(f"Factor de corrección: {factor_renormalizacion}")

Pk_final = Pk * factor_renormalizacion
ruta_final = 'Analysis/Pr_dense_input_NORMALIZADO2.dat'
np.savetxt(ruta_final, np.column_stack([k, Pk_final]))
