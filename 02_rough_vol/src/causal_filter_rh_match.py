"""
causal_filter_rh_match.py  —  close the rough-Heston cross-model bias
by matching the *V* (not log-V, not fGn) PSD directly.

Hypothesis: the residual −0.15 bias in §5.4's cross-model experiment
comes from the V-distribution mismatch (log-normal rBergomi V vs
mean-reverting rough Heston V).  Matching the training PSD to rough
Heston's V process — and reconstructing V linearly as
   V_surr = θ + filter(z)
instead of exponentially — should narrow that gap.

This is a Gaussian linearisation of rough Heston: it captures
second-order statistics (autocovariance) but not higher-order
(positivity, skewness).  How much of the leverage Greek bias is
explained by the Gaussian second-order law alone?
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
N_PSD_MC                = 30_000      # paths for estimating V PSD
EPOCHS                  = 5_000       # more epochs since RL init is wrong for V
V_MIN                   = 1.0e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  Rough Heston reference (8k paths, ρ=−0.9) ────────────────────────────
print("--- Rough Heston reference (ρ=−0.9) ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)
print(f"  E[V_rh] = {V_rh.mean():.5f}  Var[V_rh] = {V_rh.var():.5f}")

iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)


# ─── 2.  PSD of (V_rh - θ) used as training target ────────────────────────────
print(f"\n--- Estimating rough Heston V PSD (MC: {N_PSD_MC:,} paths) ---")
V_mc, _, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_PSD_MC, T, seed=42)
X = V_mc[:, 1:] - THETA            # centered V, drop t=0 (deterministic V_0)
freqs, psd_rh = compute_psd(X, DT)
print(f"  PSD: low={psd_rh[:5].mean():.3e}  high={psd_rh[-5:].mean():.3e}")
print(f"  std(V_mc) = {V_mc.std():.5f}")


# ─── 3.  Train CausalSpectralFilter on the rough Heston V PSD ─────────────────
print("\n--- Training Causal SF on rough Heston V PSD ---")
model = CausalSpectralFilter(N, H=H, dt=DT)
# Rescale RL init so initial output variance roughly matches target
with torch.no_grad():
    target_var = float(np.mean(psd_rh) * 2 / DT / N) * N          # heuristic
    init_var   = float((model.h ** 2).sum().item())
    if init_var > 0:
        model.h.data *= np.sqrt(target_var / init_var) * 0.5      # 0.5: undershoot for safety
model, _ = train_one(H, N, DT, psd_rh, 'Causal-RH-V',
                     n_epochs=EPOCHS, device=device, model=model)


# ─── 4.  Generate surrogate paths: V_surr = θ + filter(z) ─────────────────────
print("\n--- Surrogate V paths (linear reconstruction) ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_t   = torch.randn(N_SURR, N, device=device)
    dW    = model(z_t).cpu().numpy()
    z_np  = z_t.cpu().numpy()

# V_surr = θ + filter(z), clipped to ≥ V_MIN
V_surr           = np.zeros((N_SURR, N + 1))
V_surr[:, 0]     = V0
V_surr[:, 1:]    = np.maximum(V_MIN, THETA + dW)
print(f"  E[V_surr] = {V_surr.mean():.5f}  (target θ = {THETA:.4f})")
print(f"  Var[V_surr] = {V_surr.var():.5f}  (rough Heston: {V_rh.var():.5f})")
print(f"  Min V_surr clipped: {(THETA + dW < V_MIN).mean()*100:.2f}% of cells")


# ─── 5.  Compute IV + delta with causal z-Cholesky ────────────────────────────
print("\n--- IV / delta with causal z-Cholesky ρ=−0.9 ---")
iv_J, _, _ = compute_iv_surface(
    V_surr, DT, innovations=z_np, rho=RHO, seed=20)
iv_J0, _, _ = compute_iv_surface(
    V_surr, DT, innovations=None, rho=0.0, seed=30)

delta_J  = compute_delta_surface(V_surr, z_np, DT, rho=RHO, seed=21)
delta_J0 = compute_delta_surface(V_surr, z_np, DT, rho=0.0, seed=31)

# ─── 6.  Summary ─────────────────────────────────────────────────────────────
def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan

j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
print("\n" + "=" * 84)
print("    JOINT (V,S):  causal SF trained on rough-Heston V PSD, linear V = θ + filter(z)")
print("=" * 84)
print(f"{'Condition':<46} | {'IV RMSE (vp)':>12} | {'ATM Δ (T=.50)':>13} | {'bias':>8}")
print("-" * 84)
ref50 = delta_ref[1, j_atm]
print(f"{'rough Heston reference (ρ=−0.9)':<46} | {' —':>12} | {ref50:13.4f} | {' —':>8}")
print(f"{'§5.3 Surr A (perfect-V floor, fresh BM)':<46} | {' —':>12} | "
      f"{ref50-0.258:13.4f} | {-0.258:+8.4f}")
print(f"{'§5.4 Causal-rBergomi z-Chol (Surr D)':<46} | {4.57:>12.2f} | {0.6247:13.4f} | {-0.150:+8.4f}")
print("-" * 84)
for label, iv_s, d in [
    ('NEW Joint-V baseline (ρ=0)',                iv_J0, delta_J0),
    ('NEW Joint-V causal z-Cholesky (ρ=−0.9)',    iv_J,  delta_J),
]:
    rmse = rmse_vp(iv_ref, iv_s)
    dval = d[1, j_atm]
    print(f"{label:<46} | {rmse:>12.2f} | {dval:13.4f} | {dval-ref50:+8.4f}")
print("=" * 84)

print("\n--- Per-maturity ATM deltas (Joint-V z-Cholesky) ---")
print(f"{'T':>6}  {'Δ_ref':>8}  {'Δ_J':>8}  {'bias':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    r  = delta_ref[i, j_atm]
    dJ = delta_J[i, j_atm]
    print(f"  {Tv:.2f}  {r:8.4f}  {dJ:8.4f}  {dJ-r:+8.4f}")

np.savez(RESULTS / 'causal_filter_rh_match.npz',
         iv_ref=iv_ref, iv_J=iv_J, iv_J0=iv_J0,
         delta_ref=delta_ref, delta_J=delta_J, delta_J0=delta_J0,
         psd_rh=psd_rh, V_surr_sample=V_surr[:50])
print(f"\nSaved → {RESULTS / 'causal_filter_rh_match.npz'}")
