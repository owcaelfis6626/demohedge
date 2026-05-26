# Spectral PINN for Rough Volatility

Fast neural surrogate for the rough Bergomi model. Train a 126-parameter
spectral filter to match the exact power spectral density of fractional
Gaussian noise, then price options at 100× the speed of Monte Carlo.

## Quick Start

```bash
pip install -r 02_rough_vol_paper/requirements_demo.txt
streamlit run 02_rough_vol_paper/streamlit_app.py
```

## What's Inside

- **SpectralFilter** (126 params): circular FFT, ρ=0 only, 1.33 vp IV RMSE
- **CausalSpectralFilter** (252 params): lower-triangular Toeplitz, ρ≠0 via z-Cholesky, 2.57 vp
- **BochnerMLP** (41k params): architecture-agnostic spectral loss, 0.61 vp at mild vol-of-vol
- **VolterraSurrogate** (255 params): autoregressive with √V_j diffusion, 0.80 vp cross-model, MC noise

## Papers

1. *Closing the Variance Gap: Bochner PINN for Lévy-Driven SPDEs* — W₂ convergence theorem, 9 benchmarks
2. *Distributional Learning of Rough Volatility via Spectral Surrogates* — Cramér PINN, causal filter, live SPY calibration

## Author

Hubert Lipski
