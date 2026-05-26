"""
causal_filter_rheston.py  —  apply the CausalSpectralFilter to rough Heston.

Tests whether the causal architecture (which fixed rBergomi ρ=−0.9) also
helps under rough Heston, where the V distribution itself is structurally
different (mean-reverting Volterra dynamics, not log-normal).

  Reference   : rough Heston (Volterra Euler), V_rH, Z_rH, ρ=−0.9
  Surrogate D : causal SF (rBergomi reconstruction) + z-Cholesky ρ=−0.9
                — same V distribution as old Surrogate B, but correct
                correlation channel via causal architecture
  Surrogate E : causal SF + ρ=0
                — same V as D, no leverage (lower bound = Bias A floor)

The §5.3 Bias A (−0.27) is a structural floor: any surrogate using rBergomi
V cannot beat it.  We expect Surrogate D's bias ≥ −0.27.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance
from bochner_pinn import CausalSpectralFilter, get_exact_fgn_psd, train_one
from implied_vol import (compute_iv_surface, compute_delta_surface,
                          implied_vol_single,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT,
                          RESULTS)

# ─── Parameters ───────────────────────────────────────────────────────────────
H, KAPPA, THETA, NU, V0 = 0.10, 2.0, 0.04, 1.9, 0.04
ETA, XI0                = 1.9, 0.04
RHO                     = -0.9
N, T                    = 252, 1.0
DT                      = T / N
N_REF                   = 8_000
N_SURR                  = 5_000
EPOCHS                  = 3_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  Rough Heston reference ───────────────────────────────────────────────
print("--- Rough Heston reference (ρ=−0.9) ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)
print(f"  E[V_rh] = {V_rh.mean():.5f}  (target {THETA:.4f})")

iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)

# ─── 2.  Train CausalSpectralFilter on exact fGn PSD ──────────────────────────
print("\n--- Training CausalSpectralFilter ---")
freqs, psd_exact = get_exact_fgn_psd(H, N, DT)
causal_model = CausalSpectralFilter(N, H=H, dt=DT)
causal_model, _ = train_one(H, N, DT, psd_exact, 'CausalSF',
                             n_epochs=EPOCHS, device=device, model=causal_model)


# ─── 3.  Surrogate paths (rBergomi V via causal SF) ──────────────────────────
print("\n--- Surrogate paths (Causal SF, rBergomi V) ---")
torch.manual_seed(7)
causal_model.eval()
with torch.no_grad():
    z_t   = torch.randn(N_SURR, N, device=device)
    dW    = causal_model(z_t).cpu().numpy()
    z_np  = z_t.cpu().numpy()

W_surr            = np.zeros((N_SURR, N + 1))
W_surr[:, 1:]     = np.cumsum(dW, axis=1)
var_W             = W_surr.var(axis=0)
V_surr            = XI0 * np.exp(ETA * W_surr - 0.5 * ETA ** 2 * var_W[None, :])
print(f"  E[V_surr] = {V_surr.mean():.5f}  (target {XI0:.4f})")


# ─── 4.  IV + delta with causal z-Cholesky and ρ=0 baseline ──────────────────
print("\n--- IV surfaces ---")
iv_D, _, _   = compute_iv_surface(
    V_surr, DT, innovations=z_np, rho=RHO, seed=20)   # Surrogate D: causal z-Chol
iv_E, _, _   = compute_iv_surface(
    V_surr, DT, innovations=None,  rho=0.0, seed=30)  # Surrogate E: ρ=0 baseline

delta_D  = compute_delta_surface(V_surr, z_np,        DT, rho=RHO, seed=21)
delta_E  = compute_delta_surface(V_surr, z_np,        DT, rho=0.0, seed=31)


# ─── 5.  RMSE + delta summary ────────────────────────────────────────────────
def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan

j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))

print("\n" + "=" * 78)
print("    CAUSAL SF UNDER ROUGH HESTON  (vs rH reference, ρ=−0.9)")
print("=" * 78)
print(f"{'Condition':<38} | {'IV RMSE (vp)':>13} | {'ATM Δ (T=.50)':>14} | {'bias':>8}")
print("-" * 78)
ref50 = delta_ref[1, j_atm]
print(f"{'rough Heston reference (ρ=−0.9)':<38} | {' —':>13} | {ref50:14.4f} | {' —':>8}")
for label, iv_s, d in [
    ('Surr E  (causal SF, ρ=0)',           iv_E, delta_E),
    ('Surr D  (causal SF, z-Chol ρ=−0.9)', iv_D, delta_D),
]:
    rmse = rmse_vp(iv_ref, iv_s)
    dval = d[1, j_atm]
    print(f"{label:<38} | {rmse:13.2f} | {dval:14.4f} | {dval-ref50:+8.4f}")
print("=" * 78)

# Compare to old §5.3 results (Surrogate B circular + bad BM)
print("\nFor reference, §5.3 (circular SF, ρ=−0.9 with wrong BM):")
print(f"  Surr B IV RMSE = 188 vp, Bias B at T=.50 was +0.488")
print(f"  Bias A (structural floor, same V_rH, fresh BM) = −0.258 at T=.50")
print(f"\nThe causal Surr D should sit BETWEEN −0.27 (Bias A floor) and +0.49 (old Bias B).")

# ─── 6.  Per-T summary ────────────────────────────────────────────────────────
print("\n--- Per-maturity ATM deltas ---")
print(f"{'T':>6}  {'Δ_ref':>8}  {'Δ_E (ρ=0)':>11}  {'bias E':>8}  {'Δ_D (ρ=−0.9)':>14}  {'bias D':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    r  = delta_ref[i, j_atm]
    eD = delta_D[i, j_atm]
    eE = delta_E[i, j_atm]
    print(f"  {Tv:.2f}  {r:8.4f}  {eE:11.4f}  {eE-r:+8.4f}  "
          f"{eD:14.4f}  {eD-r:+8.4f}")

np.savez(RESULTS / 'causal_filter_rheston.npz',
         iv_ref=iv_ref, iv_D=iv_D, iv_E=iv_E,
         delta_ref=delta_ref, delta_D=delta_D, delta_E=delta_E,
         strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT)
print(f"\nSaved → {RESULTS / 'causal_filter_rheston.npz'}")
