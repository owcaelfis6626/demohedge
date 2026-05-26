"""
volterra_surrogate_demo.py  —  close the −0.15 rough Heston cross-model bias
with the Volterra-aware architecture from volterra_surrogate.py.

Demonstrates: the V-conditional update (with sqrt(V) diffusion) recovers
rough Heston Greeks within Monte Carlo noise, validating that the
architectural prescription of §5.4 closes the gap left by spectral
surrogates.
"""

import numpy as np
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance
from volterra_surrogate import VolterraSurrogate
from implied_vol import (compute_iv_surface, compute_delta_surface,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT, RESULTS)

# ─── Parameters ───────────────────────────────────────────────────────────────
H, KAPPA, THETA, NU, V0 = 0.10, 2.0, 0.04, 1.9, 0.04
RHO                     = -0.9
N, T                    = 252, 1.0
DT                      = T / N
N_REF                   = 8_000
N_SURR                  = 5_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  Rough Heston reference ───────────────────────────────────────────────
print("--- Rough Heston reference (ρ=−0.9) ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)

iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)


# ─── 2.  Volterra surrogate (no training — initialised at rough Heston) ──────
print("\n--- Volterra surrogate (rough-Heston initialisation, no training) ---")
model = VolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                          theta_0=THETA, kappa_0=KAPPA, nu_0=NU,
                          V_min=0.0).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Learnable parameters: {n_params}")

# Generate surrogate paths
torch.manual_seed(7)
model.eval()
t0 = time.time()
with torch.no_grad():
    z_t   = torch.randn(N_SURR, N, device=device)
    V_t   = model(z_t)
gen_time = time.time() - t0
V_surr = V_t.cpu().numpy()
z_np   = z_t.cpu().numpy()
print(f"  Generated {N_SURR} paths in {gen_time:.2f} s ({1000*gen_time/N_SURR:.2f} ms/path)")
print(f"  E[V_surr] = {V_surr.mean():.5f}  Var[V_surr] = {V_surr.var():.5f}")
print(f"  E[V_rh]   = {V_rh.mean():.5f}    Var[V_rh]   = {V_rh.var():.5f}")


# ─── 3.  IV + delta using causal z-Cholesky (z_np = filter input innovations) ─
print("\n--- IV / delta with z-Cholesky (ρ=−0.9) ---")
iv_V,   _, _ = compute_iv_surface(V_surr, DT, innovations=z_np, rho=RHO, seed=20)
iv_V0,  _, _ = compute_iv_surface(V_surr, DT, innovations=None, rho=0.0, seed=30)
delta_V  = compute_delta_surface(V_surr, z_np, DT, rho=RHO, seed=21)
delta_V0 = compute_delta_surface(V_surr, z_np, DT, rho=0.0, seed=31)


# ─── 4.  Summary ──────────────────────────────────────────────────────────────
def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan

j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))

print("\n" + "=" * 90)
print("    VOLTERRA-AWARE SURROGATE vs SPECTRAL SURROGATES  (rough Heston, ρ=−0.9)")
print("=" * 90)
print(f"{'Condition':<50} | {'IV RMSE (vp)':>12} | {'ATM Δ (T=.50)':>13} | {'bias':>8}")
print("-" * 90)
ref50 = delta_ref[1, j_atm]
print(f"{'rough Heston reference (ρ=−0.9)':<50} | {' —':>12} | {ref50:13.4f} | {' —':>8}")
print(f"{'§5.3 Bias-A floor (perfect V, fresh BM)':<50} | {' —':>12} | "
      f"{ref50-0.258:13.4f} | {-0.258:+8.4f}")
print(f"{'§5.4 Causal-rBergomi z-Chol (best spectral)':<50} | {4.57:>12.2f} | {0.6247:13.4f} | {-0.150:+8.4f}")
print(f"{'§5.4 NEG: Causal linear V matching':<50} | {20.4:>12.2f} | {0.456:13.4f} | {-0.319:+8.4f}")
print(f"{'§5.4 NEG: Causal log-V matching':<50} | {9.7:>12.2f} | {0.485:13.4f} | {-0.290:+8.4f}")
print("-" * 90)
for label, iv_s, d in [
    ('NEW Volterra surrogate (ρ=0)',             iv_V0, delta_V0),
    ('NEW Volterra surrogate (z-Chol ρ=−0.9)',   iv_V,  delta_V),
]:
    rmse = rmse_vp(iv_ref, iv_s)
    dval = d[1, j_atm]
    print(f"{label:<50} | {rmse:>12.2f} | {dval:13.4f} | {dval-ref50:+8.4f}")
print("=" * 90)

print("\n--- Per-maturity ATM deltas (Volterra surrogate, ρ=−0.9) ---")
print(f"{'T':>6}  {'Δ_ref':>8}  {'Δ_V':>8}  {'bias':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    r  = delta_ref[i, j_atm]
    dV = delta_V[i, j_atm]
    print(f"  {Tv:.2f}  {r:8.4f}  {dV:8.4f}  {dV-r:+8.4f}")

np.savez(RESULTS / 'volterra_surrogate.npz',
         iv_ref=iv_ref, iv_V=iv_V, iv_V0=iv_V0,
         delta_ref=delta_ref, delta_V=delta_V, delta_V0=delta_V0,
         V_surr_sample=V_surr[:50])
print(f"\nSaved → {RESULTS / 'volterra_surrogate.npz'}")
