# Higgs-R^2 Inflation

A Python package for computing the inflationary background and the evolution of primordial perturbations in the multi-field Higgs–R² inflation model.

## Overview

This code provides a framework to analyze the multi-field dynamics of the Higgs–R² inflation model as a function of the number of e-folds or cosmic time. It was developed for the paper:

> **"Primordial features and low-ℓ suppression from isocurvature modes in multi-field Higgs-R² inflation"**
> *F. Pineda, L. O. Pimentel (2025)*
> [arXiv:2512.xxxxx](https://arxiv.org/abs/2512.xxxxx) (Update this link when available)

The code automatically handles the following:
- Solving the background dynamics.
- Computing adiabatic, isocurvature, cross-correlation, and tensor perturbations.
- Determining the primordial power spectra.
- Computing CMB observables.

The codebase is written in Python with a modular, object-oriented design (OOP), making it easy to extend or adapt to other inflationary models.

## Disclaimer

This software was developed for academic research purposes. While efforts have been made to ensure accuracy and reproducibility, the code is provided "as is". We have included Jupyter notebooks in the `examples/` folder to demonstrate the main functionality and reproduce the figures from the paper.

## Installation

Since this is a research code, we recommend installing it in **editable mode**. This allows you to modify the source code or notebooks and see changes immediately.

### Prerequisites
- Python 3.8+
- git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TU_USUARIO/Higgs-R-2-inflation.git
   cd Higgs-R-2-inflation

2. **Install the package:**
   ```bash
    pip install -e .