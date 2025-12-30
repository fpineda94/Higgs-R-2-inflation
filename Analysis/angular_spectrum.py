import numpy as np
import matplotlib.pyplot as plt
from classy import Class


common_params = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'tau_reio': 0.0544,
    'output': 'tCl,pCl,lCl',
    'lensing': 'yes',
    'l_max_scalars': 2500,
}


ruta_archivo = '/Users/flaviopineda/Documents/Fisica/Proyectos/Generating isocurvature modes and primordial features in multi-field Higgs-R^2 inflation/Analysis/Pr_dense_input_xi_0.1.dat'

params_model = common_params.copy()
params_model.update({
    'P_k_ini type': 'external_Pk',
    'command': f"cat '{ruta_archivo}'"
})

print("Calculando Higgs-R^2...")
cosmo_model = Class()
cosmo_model.set(params_model)
cosmo_model.compute()
cls_model = cosmo_model.lensed_cl(2500)
cosmo_model.struct_cleanup() 

params_ref = common_params.copy()
params_ref.update({
    'A_s': 2.1e-9,   
    'n_s': 0.9649    
})

print("Calculando LambdaCDM de referencia...")
cosmo_ref = Class()
cosmo_ref.set(params_ref)
cosmo_ref.compute()
cls_ref = cosmo_ref.lensed_cl(2500)
cosmo_ref.struct_cleanup()

ll = cls_ref['ell'][2:] 

Tcmb = 2.7255e6 
factor = ll * (ll + 1) / (2 * np.pi) * (Tcmb**2)

clTT_model = cls_model['tt'][2:] * factor
clTT_ref   = cls_ref['tt'][2:]   * factor

diff = (clTT_model - clTT_ref) / clTT_ref

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(ll, clTT_ref, 'k--', label=r'$\Lambda$CDM (Planck 18)', alpha=0.7)
ax1.plot(ll, clTT_model, 'b-', label=r'Higgs-$R^2$ (Tu modelo)', linewidth=2)
ax1.set_ylabel(r'$\mathcal{D}_\ell^{TT} [\mu K^2]$', fontsize=12)
ax1.set_title(r'Comparación Directa: Higgs-$R^2$ vs Estándar', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.3)
ax1.set_xscale('log') 
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.plot(ll, diff, 'r-', linewidth=1.5)
ax2.set_ylabel(r'$\Delta C_\ell / C_\ell^{ref}$', fontsize=12)
ax2.set_xlabel(r'Multipolo $\ell$', fontsize=12)
ax2.set_ylim(-0.2, 0.2) 
ax2.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()