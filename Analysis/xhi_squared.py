import numpy as np
import matplotlib.pyplot as plt
from classy import Class

# --- 1. CONFIGURACIÓN ---
common_params = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'tau_reio': 0.0544,
    
    # --- AQUÍ ESTÁ EL CAMBIO ---
    # Antes tenías: 'output': 'tCl', 
    # Ahora debe ser:
    'output': 'tCl,lCl',  # <--- Agrega ',lCl'
    
    'lensing': 'yes',
    'l_max_scalars': 2500,
}

# --- 2. GENERAR "DATOS" (Referencia LambdaCDM) ---
# Asumimos que Planck coincide con esto
print("Generando referencia (LambdaCDM)...")
params_ref = common_params.copy()
params_ref.update({'A_s': 2.1e-9, 'n_s': 0.9649})
cosmo_ref = Class()
cosmo_ref.set(params_ref)
cosmo_ref.compute()
cls_ref = cosmo_ref.lensed_cl(2500)

# Extraemos Cls SIN el factor l(l+1) para el cálculo estadístico
# (El chi2 se calcula sobre los C_l crudos)
ll = cls_ref['ell'][2:]
cl_tt_data = cls_ref['tt'][2:]

# --- 3. TU MODELO (El Acusado) ---
# Pon aquí tu archivo .dat actual (el que dices que "no queda")
ruta_tu_archivo = '/Users/flaviopineda/Documents/Fisica/Proyectos/Generating isocurvature modes and primordial features in multi-field Higgs-R^2 inflation/Analysis/Pr_dense_input_xi_0.1.dat'

print("Calculando tu modelo...")
params_model = common_params.copy()
params_model.update({
    'P_k_ini type': 'external_Pk',
    'command': f"cat '{ruta_tu_archivo}'"
})
cosmo = Class()
cosmo.set(params_model)
cosmo.compute()
cls_model = cosmo.lensed_cl(2500)
cl_tt_model = cls_model['tt'][2:]

# --- 4. CÁLCULO DEL CHI-CUADRADO ---
# Definimos el error (Sigma).
# A falta de datos reales de Planck aquí, usamos la VARIANZA CÓSMICA pura.
# Es el límite teórico de precisión: sigma_l = Cl * sqrt(2 / (2l + 1))
# f_sky (fracción del cielo) ~ 0.6 para Planck conservador
f_sky = 0.6
sigma_l = cl_tt_data * np.sqrt(2 / ((2 * ll + 1) * f_sky))

# Fórmula del Chi^2
# Chi2 = Suma [ (Model - Data)^2 / Sigma^2 ]
diferencia_cuadrada = (cl_tt_model - cl_tt_data)**2
chi2_por_l = diferencia_cuadrada / (sigma_l**2)

# Calculamos en dos rangos
# Rango Total
chi2_total = np.sum(chi2_por_l)
dof_total = len(ll) # Grados de libertad

# Rango Problemático (High-l, l > 800)
mask_high = ll > 800
chi2_high = np.sum(chi2_por_l[mask_high])
dof_high = np.sum(mask_high)

# Chi2 Reducido (Debería ser cercano a 1 si el ajuste es bueno)
chi2_red_total = chi2_total / dof_total
chi2_red_high = chi2_high / dof_high

print("\n" + "="*40)
print("RESULTADOS DEL DIAGNÓSTICO")
print("="*40)
print(f"Chi^2 Total ({len(ll)} puntos): {chi2_total:.2f}")
print(f"Chi^2 Reducido Total:         {chi2_red_total:.2f}  <-- (Objetivo: ~1.0)")
print("-" * 40)
print(f"Chi^2 en cola (l > 800):      {chi2_high:.2f}")
print(f"Chi^2 Reducido (l > 800):     {chi2_red_high:.2f}  <-- (Aquí te mató el editor)")
print("="*40)

# Interpretación Automática
if chi2_red_high > 2.0:
    print(">>> ALERTA ROJA: Tu modelo está a más de 2-sigma de los datos en escalas pequeñas.")
    print(">>> CAUSA PROBABLE: El índice espectral (tilt) es incorrecto.")
    print(">>> SOLUCIÓN: Necesitas aumentar N_efolds (ir de 50 a 55 o 60).")
elif chi2_red_high > 1.2:
    print(">>> ALERTA AMARILLA: El ajuste es pobre pero quizás defendible con 'tensión'.")
    print(">>> Intenta ajustar H_0 en CLASS ligeramente.")
else:
    print(">>> SEMÁFORO VERDE: El ajuste es estadísticamente aceptable.")

# --- 5. VISUALIZACIÓN DE DESVIACIONES ---
plt.figure(figsize=(10, 5))
# Graficamos la desviación en unidades de Sigma (Barras de error)
desviacion_sigma = (cl_tt_model - cl_tt_data) / sigma_l

plt.plot(ll, desviacion_sigma, color='red', alpha=0.7, label=r'Desviación ($\sigma$)')
plt.axhline(1, color='k', linestyle=':', alpha=0.5)
plt.axhline(-1, color='k', linestyle=':', alpha=0.5)
plt.axhline(0, color='k', linestyle='-')
plt.fill_between(ll, -1, 1, color='gray', alpha=0.2, label='Zona Segura (1$\sigma$)')
plt.fill_between(ll, -2, 2, color='gray', alpha=0.1, label='Zona Límite (2$\sigma$)')

plt.xlabel(r'Multipolo $\ell$')
plt.ylabel(r'Desviación en $\sigma$ [(Model-Data)/Error]')
plt.title('Diagnóstico de Rechazo: ¿Qué tan lejos estás?')
plt.legend()
plt.ylim(-5, 5) # Zoom en la zona relevante
plt.xlim(2, 2500)
plt.xscale('log')
plt.grid(alpha=0.3)
plt.show()

cosmo.struct_cleanup()
cosmo_ref.struct_cleanup()