"""
markovian_volterra_demo.py  —  sub-quadratic Volterra surrogate vs the
O(N^2) original, and vs rough Heston reference.

Sweep over M ∈ {3, 5, 7, 10} exponential modes.  Show the speed/accuracy
tradeoff: at M = 10, IV RMSE ≤ a few vp and the per-path cost drops from
O(N^2) to O(M N) — a ~25× reduction at N = 252.
"""

import numpy as np
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance
from volterra_surrogate import VolterraSurrogate
from markovian_volterra import MarkovianVolterraSurrogate, fit_rl_exponentials
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


# ─── 1.  RL kernel fit diagnostics ────────────────────────────────────────────
print("--- Exponential-sum approximation of the RL kernel ---")
for M in (3, 5, 7, 10):
    _, _, err = fit_rl_exponentials(N, H, M=M)
    print(f"  M={M:>2}: relative L2 fit error = {err:.4e}")


# ─── 2.  Rough Heston reference ───────────────────────────────────────────────
print("\n--- Rough Heston reference (ρ=−0.9) ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)
iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)
print(f"  E[V_rh]   = {V_rh.mean():.5f}")


# ─── 3.  Sweep over M ────────────────────────────────────────────────────────
def eval_surrogate(model, label):
    """Generate 5k surrogate paths, compute IV / delta vs rough Heston."""
    torch.manual_seed(7)
    model.eval()
    # Time the forward pass
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        z_t = torch.randn(N_SURR, N, device=device)
        V_t = model(z_t)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fwd_ms = 1e3 * (time.time() - t0)
    V_surr = V_t.cpu().numpy()
    z_np   = z_t.cpu().numpy()

    iv,   _, _ = compute_iv_surface(V_surr, DT, innovations=z_np, rho=RHO, seed=20)
    delta      = compute_delta_surface(V_surr, z_np, DT, rho=RHO, seed=21)

    diff = (iv - iv_ref) * 100
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
    bias_T05 = delta[1, j_atm] - delta_ref[1, j_atm]

    print(f"  {label:<32}  fwd {fwd_ms:7.1f} ms   IV RMSE {rmse:5.2f} vp   "
          f"ATM Δ T=.50 bias {bias_T05:+.4f}")
    return rmse, bias_T05, fwd_ms


# Baseline: O(N^2) Volterra surrogate (architecturally exact)
print("\n--- Baseline: O(N^2) VolterraSurrogate (rough-Heston init) ---")
model_full = VolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                                theta_0=THETA, kappa_0=KAPPA, nu_0=NU,
                                V_min=1e-6).to(device)
rmse_full, bias_full, time_full = eval_surrogate(model_full, "O(N^2) Volterra")

print("\n--- Sweep: Markovian Volterra surrogates (M modes) ---")
sweep = {}
for M in (3, 5, 7, 10):
    model_m = MarkovianVolterraSurrogate(
        N=N, dt=DT, H=H, M=M, V_0=V0,
        theta_0=THETA, kappa_0=KAPPA, nu_0=NU, V_min=1e-6).to(device)
    rmse_m, bias_m, time_m = eval_surrogate(model_m, f"Markovian (M={M:>2})")
    sweep[M] = dict(rmse=rmse_m, bias=bias_m, time_ms=time_m,
                    fit_err=model_m.fit_rel_err)


# ─── 4.  Summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("    SUB-QUADRATIC VOLTERRA SURROGATE  vs  O(N^2) BASELINE")
print("=" * 78)
print(f"{'Method':<26}  {'Fwd (ms)':>10}  {'Speedup':>9}  "
      f"{'IV RMSE':>9}  {'ATM Δ bias':>11}")
print("-" * 78)
print(f"{'O(N^2) Volterra (exact)':<26}  {time_full:>10.1f}  {'1.0x':>9}  "
      f"{rmse_full:>7.2f}vp  {bias_full:>+11.4f}")
for M in (3, 5, 7, 10):
    r = sweep[M]
    speedup = time_full / r['time_ms']
    print(f"{f'Markovian (M={M:>2})':<26}  {r['time_ms']:>10.1f}  {speedup:>8.1f}x  "
          f"{r['rmse']:>7.2f}vp  {r['bias']:>+11.4f}")
print("=" * 78)
print(f"(All evaluated at ρ=−0.9 vs rough Heston reference, 5k surrogate paths.)")

np.savez(RESULTS / 'markovian_volterra_sweep.npz',
         M_values=np.array(list(sweep.keys())),
         rmse=np.array([sweep[M]['rmse'] for M in sweep]),
         bias=np.array([sweep[M]['bias'] for M in sweep]),
         time_ms=np.array([sweep[M]['time_ms'] for M in sweep]),
         fit_err=np.array([sweep[M]['fit_err'] for M in sweep]),
         rmse_full=rmse_full, bias_full=bias_full, time_full=time_full)
print(f"\nSaved → {RESULTS / 'markovian_volterra_sweep.npz'}")
