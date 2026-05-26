"""
causal_filter_rh_logV.py  —  log-V matching attempt for rough Heston.

Train CausalSpectralFilter to match the PSD of log(V_rh / theta), then
reconstruct V_surr = theta * exp(W - 0.5 * Var(W)) where W is the cumsum
of the filter output.  This is "rBergomi with rough-Heston-calibrated PSD"
- a log-normal surrogate whose autocovariance matches rough Heston log-V.

Earlier circular-filter attempt collapsed (E[V] -> 0).  Question: does
causal architecture + careful variance correction help?
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance, compute_psd
from bochner_pinn import CausalSpectralFilter, train_one
from implied_vol import (compute_iv_surface, compute_delta_surface,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT, RESULTS)

# ─── Parameters ───────────────────────────────────────────────────────────────
H, KAPPA, THETA, NU, V0 = 0.10, 2.0, 0.04, 1.9, 0.04
RHO                     = -0.9
N, T                    = 252, 1.0
DT                      = T / N
N_REF                   = 8_000
N_SURR                  = 5_000
N_PSD_MC                = 30_000
EPOCHS                  = 5_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  Rough Heston reference ───────────────────────────────────────────────
print("--- Rough Heston reference ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)

iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)


# ─── 2.  PSD of log-V INCREMENTS of rough Heston ──────────────────────────────
print(f"\n--- log-V increment PSD (MC: {N_PSD_MC:,} paths) ---")
V_mc, _, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_PSD_MC, T, seed=42)
# Use increments of log V (analogous to fGn for rBergomi)
log_V = np.log(np.clip(V_mc, 1e-12, None))
d_log_V = np.diff(log_V, axis=1)   # (n_paths, N) increments of log V
freqs, psd_logV = compute_psd(d_log_V, DT)
print(f"  PSD: low={psd_logV[:5].mean():.3e}  high={psd_logV[-5:].mean():.3e}")
print(f"  Var(d log V) = {d_log_V.var():.5f}")
print(f"  Var(log V_T) = {log_V[:, -1].var():.5f}")


# ─── 3.  Train CausalSpectralFilter ───────────────────────────────────────────
print("\n--- Training causal filter on log-V increment PSD ---")
model = CausalSpectralFilter(N, H=H, dt=DT)
model, _ = train_one(H, N, DT, psd_logV, 'Causal-logV-rH',
                     n_epochs=EPOCHS, device=device, model=model)


# ─── 4.  Surrogate V via exp reconstruction ──────────────────────────────────
print("\n--- Surrogate V (exp reconstruction) ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_t   = torch.randn(N_SURR, N, device=device)
    dW    = model(z_t).cpu().numpy()
    z_np  = z_t.cpu().numpy()

W            = np.zeros((N_SURR, N + 1))
W[:, 1:]     = np.cumsum(dW, axis=1)
var_W        = W.var(axis=0)
V_surr       = THETA * np.exp(W - 0.5 * var_W[None, :])
print(f"  E[V_surr] = {V_surr.mean():.5f}  Var[V_surr] = {V_surr.var():.5f}")
print(f"  E[V_rh]   = {V_rh.mean():.5f}    Var[V_rh]   = {V_rh.var():.5f}")


# ─── 5.  IV + delta ──────────────────────────────────────────────────────────
print("\n--- IV / delta ---")
iv_L,   _, _ = compute_iv_surface(V_surr, DT, innovations=z_np, rho=RHO, seed=20)
iv_L0,  _, _ = compute_iv_surface(V_surr, DT, innovations=None, rho=0.0, seed=30)
delta_L  = compute_delta_surface(V_surr, z_np, DT, rho=RHO, seed=21)
delta_L0 = compute_delta_surface(V_surr, z_np, DT, rho=0.0, seed=31)


# ─── 6.  Summary ─────────────────────────────────────────────────────────────
def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan

j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
print("\n" + "=" * 88)
print("    LOG-V MATCHING:  causal SF, rough-Heston log-V PSD, exp reconstruction")
print("=" * 88)
print(f"{'Condition':<48} | {'IV RMSE (vp)':>12} | {'ATM Δ (T=.50)':>13} | {'bias':>8}")
print("-" * 88)
ref50 = delta_ref[1, j_atm]
print(f"{'rough Heston reference (ρ=−0.9)':<48} | {' —':>12} | {ref50:13.4f} | {' —':>8}")
print(f"{'§5.3 Bias-A floor (perfect V, fresh BM)':<48} | {' —':>12} | "
      f"{ref50-0.258:13.4f} | {-0.258:+8.4f}")
print(f"{'§5.4 Causal-rBergomi z-Chol (Surr D)':<48} | {4.57:>12.2f} | {0.6247:13.4f} | {-0.150:+8.4f}")
print("-" * 88)
for label, iv_s, d in [
    ('NEW Causal-logV baseline (ρ=0)',                iv_L0, delta_L0),
    ('NEW Causal-logV z-Cholesky (ρ=−0.9)',           iv_L,  delta_L),
]:
    rmse = rmse_vp(iv_ref, iv_s)
    dval = d[1, j_atm]
    print(f"{label:<48} | {rmse:>12.2f} | {dval:13.4f} | {dval-ref50:+8.4f}")
print("=" * 88)

print("\n--- Per-maturity ATM deltas (Causal-logV z-Cholesky) ---")
print(f"{'T':>6}  {'Δ_ref':>8}  {'Δ_L':>8}  {'bias':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    r  = delta_ref[i, j_atm]
    dL = delta_L[i, j_atm]
    print(f"  {Tv:.2f}  {r:8.4f}  {dL:8.4f}  {dL-r:+8.4f}")

np.savez(RESULTS / 'causal_filter_rh_logV.npz',
         iv_ref=iv_ref, iv_L=iv_L, iv_L0=iv_L0,
         delta_ref=delta_ref, delta_L=delta_L, delta_L0=delta_L0,
         psd_logV=psd_logV, V_surr_sample=V_surr[:50])
print(f"\nSaved → {RESULTS / 'causal_filter_rh_logV.npz'}")
