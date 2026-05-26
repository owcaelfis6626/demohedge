"""
causal_filter_hurst_sweep.py  —  CausalSpectralFilter robustness across H.

Demonstrates that the 63x IV-RMSE improvement of causal vs circular
z-Cholesky is not H=0.1 specific.  Runs all four conditions at three
Hurst values typical of rough-vol applications.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_blp
from bochner_pinn import (SpectralFilter, CausalSpectralFilter,
                           get_exact_fgn_psd, train_one)
from implied_vol import (implied_vol_single,
                         _STRIKES_DEFAULT, _MATURITIES_DEFAULT, RESULTS)

# ─── Parameters ───────────────────────────────────────────────────────────────
ETA, XI0 = 1.9, 0.04
RHO       = -0.9
N, T      = 252, 1.0
DT        = T / N
N_REF     = 8_000
N_SURR    = 5_000
EPOCHS    = 3_000
H_VALUES  = [0.05, 0.10, 0.15]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


def iv_blp(V, dZ_S, dt):
    V_left  = V[:, :-1].clip(0)
    log_inc = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ_S
    n       = V.shape[0]
    log_S   = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    S = np.exp(log_S)
    iv = np.full((len(_MATURITIES_DEFAULT), len(_STRIKES_DEFAULT)), np.nan)
    for i, Tv in enumerate(_MATURITIES_DEFAULT):
        idx = min(int(round(Tv / dt)), V.shape[1] - 1)
        S_T = S[:, idx]
        for j, K in enumerate(_STRIKES_DEFAULT):
            price = np.mean(np.maximum(S_T - K, 0.0))
            iv[i, j] = implied_vol_single(price, 1.0, K, 0.0, Tv)
    return iv


def delta_atm_blp(V, dZ_S, dt, bump=0.01):
    n      = V.shape[0]
    V_left = V[:, :-1].clip(0)
    j_atm  = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
    K_atm  = _STRIKES_DEFAULT[j_atm]

    def _price(s0v):
        log_inc = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ_S
        log_S   = np.log(s0v) + np.concatenate(
            [np.zeros((n, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        S = np.exp(log_S)
        p = np.full(len(_MATURITIES_DEFAULT), np.nan)
        for i, Tv in enumerate(_MATURITIES_DEFAULT):
            idx = min(int(round(Tv / dt)), V.shape[1] - 1)
            p[i] = np.mean(np.maximum(S[:, idx] - K_atm, 0.0))
        return p

    return (_price(1.0 + bump) - _price(1.0 - bump)) / (2.0 * bump)


def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan


def surr_pack(model, n, H_, seed_z=7, seed_perp=99):
    torch.manual_seed(seed_z)
    model.eval()
    with torch.no_grad():
        z_t  = torch.randn(n, N, device=device)
        dW   = model(z_t).cpu().numpy()
        z_np = z_t.cpu().numpy()
    W            = np.zeros((n, N + 1))
    W[:, 1:]     = np.cumsum(dW, axis=1)
    var_W        = W.var(axis=0)
    V            = XI0 * np.exp(ETA * W - 0.5 * ETA**2 * var_W[None, :])
    rng          = np.random.default_rng(seed_perp)
    z_perp       = rng.standard_normal((n, N))
    dZ_rho       = RHO * z_np + np.sqrt(1 - RHO**2) * z_perp
    dZ_rho0      = z_perp
    return V, dZ_rho, dZ_rho0


# ─── Sweep ────────────────────────────────────────────────────────────────────
results = {}
for H in H_VALUES:
    print(f"\n===========================  H = {H:.2f}  ===========================")

    # BLP reference
    V_ref, dZ_S_ref, _ = simulate_rbm_blp(H, ETA, XI0, N, N_REF, T, rho=RHO, seed=1)
    iv_ref      = iv_blp(V_ref, dZ_S_ref, DT)
    delta_ref   = delta_atm_blp(V_ref, dZ_S_ref, DT)
    print(f"  BLP ref:   E[V]={V_ref.mean():.5f}   ATM Δ(T=.5)={delta_ref[1]:.3f}")

    # Train both filters
    freqs, psd_exact = get_exact_fgn_psd(H, N, DT)
    causal_model = CausalSpectralFilter(N, H=H, dt=DT)
    causal_model, _ = train_one(H, N, DT, psd_exact, f'CausalSF H={H:.2f}',
                                 n_epochs=EPOCHS, device=device, model=causal_model)
    circ_model, _   = train_one(H, N, DT, psd_exact, f'CircularSF H={H:.2f}',
                                 n_epochs=EPOCHS, device=device)

    # Evaluate
    V_c, dZ_c_r, dZ_c_r0 = surr_pack(causal_model, N_SURR, H)
    V_f, dZ_f_r, _       = surr_pack(circ_model,   N_SURR, H)

    iv_c_r   = iv_blp(V_c, dZ_c_r,  DT)
    iv_c_r0  = iv_blp(V_c, dZ_c_r0, DT)
    iv_f_r   = iv_blp(V_f, dZ_f_r,  DT)

    d_c_r    = delta_atm_blp(V_c, dZ_c_r,  DT)
    d_c_r0   = delta_atm_blp(V_c, dZ_c_r0, DT)

    rmse_c_r  = rmse_vp(iv_ref, iv_c_r)
    rmse_c_r0 = rmse_vp(iv_ref, iv_c_r0)
    rmse_f_r  = rmse_vp(iv_ref, iv_f_r)

    print(f"\n  IV RMSE (vp) vs BLP reference:")
    print(f"    Circular SF z-Cholesky (ρ=−0.9):   {rmse_f_r:>8.2f}")
    print(f"    Causal SF   baseline   (ρ=0):      {rmse_c_r0:>8.2f}")
    print(f"    Causal SF   z-Cholesky (ρ=−0.9):   {rmse_c_r:>8.2f}")
    print(f"\n  ATM Δ at T=0.50:")
    print(f"    Reference:                           {delta_ref[1]:.4f}")
    print(f"    Causal SF baseline (ρ=0):            {d_c_r0[1]:.4f}  (bias {d_c_r0[1]-delta_ref[1]:+.4f})")
    print(f"    Causal SF z-Cholesky (ρ=−0.9):       {d_c_r[1]:.4f}  (bias {d_c_r[1]-delta_ref[1]:+.4f})")

    results[H] = dict(
        rmse_circ_rho=rmse_f_r, rmse_causal_rho0=rmse_c_r0, rmse_causal_rho=rmse_c_r,
        delta_ref=delta_ref, delta_causal_rho=d_c_r, delta_causal_rho0=d_c_r0,
        improvement=rmse_f_r / rmse_c_r if rmse_c_r > 0 else np.nan,
    )

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("                  CAUSAL FILTER ROBUSTNESS ACROSS H")
print("=" * 78)
print(f"{'H':>6} | {'Circ z-Chol':>11} | {'Causal ρ=0':>10} | {'Causal z-Chol':>13} | {'×improv.':>9} | {'ATMΔ bias':>10}")
print("-" * 78)
for H in H_VALUES:
    r = results[H]
    print(f"{H:6.2f} | {r['rmse_circ_rho']:>11.1f} | {r['rmse_causal_rho0']:>10.2f} | "
          f"{r['rmse_causal_rho']:>13.2f} | {r['improvement']:>9.1f} | "
          f"{r['delta_causal_rho'][1] - r['delta_ref'][1]:>+10.4f}")
print("=" * 78)
print("(IV RMSE in vol points vs BLP reference; ATM Δ bias at T=0.50)")

np.savez(RESULTS / 'causal_filter_hurst_sweep.npz',
         H_values=np.array(H_VALUES),
         **{f'H{H:.2f}': results[H] for H in H_VALUES})
print(f"\nSaved → {RESULTS / 'causal_filter_hurst_sweep.npz'}")
