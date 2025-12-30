import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

data = np.loadtxt("Analysis/power_spectrum_end_data.txt")
k_modes = data[:, 0]
P_R = data[:, 1]

Pr_spline = InterpolatedUnivariateSpline(k_modes, P_R, k = 3, ext = 0)

kmin = 1e-7
kmax = 5

k_class = np.logspace(np.log10(kmin), np.log10(kmax), 20000)
P_class = Pr_spline(k_class)

np.savetxt('Pr_dense_input_xi_0.1_1.dat', np.column_stack([k_class, P_class]))