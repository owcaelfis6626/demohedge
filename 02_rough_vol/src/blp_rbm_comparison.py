"""
blp_rbm_comparison.py  —  rBergomi ρ=−0.9 via BLP hybrid scheme.

Reference : RL-discretised rBergomi, stock BM correlated with driving BM Z.
Surrogate A: SpectralFilter + z-Cholesky (same z that drove the filter).
Surrogate B: SpectralFilter + ρ=0 (baseline, no leverage).

Key question: does z-based Cholesky recover the leverage-induced IV skew?
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_blp
from bochner_pinn import SpectralFilter, get_exact_fgn_psd, train_one
from implied_vol import (implied_vol_single,
                         _STRIKES_DEFAULT, _MATURITIES_DEFAULT,
                         FIGURES, RESULTS)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Parameters ───────────────────────────────────────────────────────────────
H, eta, xi0 = 0.10, 1.9, 0.04
rho          = -0.9
N, T         = 252, 1.0
dt           = T / N
N_REF        = 8_000
N_SURR       = 5_000
EPOCHS       = 3_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def iv_surface(V, dZ_S, dt, S0=1.0, r=0.0,
               strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT):
    """
    IV surface from variance V and PRE-CORRELATED stock BM increments dZ_S.
    dZ_S : (n_paths, N) unit-normal, already Cholesky-correlated with V.
    """
    V_left  = V[:, :-1].clip(0)
    log_inc = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ_S
    n       = V.shape[0]
    log_S   = np.concatenate(
        [np.full((n, 1), np.log(S0)), np.cumsum(log_inc, axis=1)], axis=1)
    S = np.exp(log_S)

    K_arr = strikes * S0
    iv    = np.full((len(maturities), len(strikes)), np.nan)
    for i, Tv in enumerate(maturities):
        idx = min(int(round(Tv / dt)), V.shape[1] - 1)
        S_T = S[:, idx]
        for j, K in enumerate(K_arr):
            price    = np.mean(np.maximum(S_T - K, 0.0)) * np.exp(-r * Tv)
            iv[i, j] = implied_vol_single(price, S0, K, r, Tv)
    return iv


def delta_surface(V, dZ_S, dt, S0=1.0, r=0.0, bump=0.01,
                  strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT):
    """FD delta; dZ_S frozen across both bumps (common random numbers)."""
    eps    = bump * S0
    n      = V.shape[0]
    V_left = V[:, :-1].clip(0)
    K_arr  = strikes * S0

    def _price(s0v):
        log_inc = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ_S
        log_S   = np.log(s0v) + np.concatenate(
            [np.zeros((n, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        S = np.exp(log_S)
        p = np.full((len(maturities), len(strikes)), np.nan)
        for i, Tv in enumerate(maturities):
            idx = min(int(round(Tv / dt)), V.shape[1] - 1)
            S_T = S[:, idx]
            for j, K in enumerate(K_arr):
                p[i, j] = np.mean(np.maximum(S_T - K, 0.0)) * np.exp(-r * Tv)
        return p

    return (_price(S0 + eps) - _price(S0 - eps)) / (2.0 * eps)


def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return np.sqrt(np.mean(d[m] ** 2))


# ─── 1.  BLP reference: rBergomi, ρ=−0.9 ─────────────────────────────────────
print("--- BLP reference (ρ=−0.9) ---")
V_ref, dZ_S_ref, t = simulate_rbm_blp(H, eta, xi0, N, N_REF, T, rho=rho, seed=1)
print(f"  E[V_ref] = {V_ref.mean():.5f}  (target {xi0:.4f})")

iv_ref    = iv_surface(V_ref, dZ_S_ref, dt)
delta_ref = delta_surface(V_ref, dZ_S_ref, dt)

print("  Reference IV (%):")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    row = "  ".join(f"{v*100:5.1f}" if np.isfinite(v) else "  nan" for v in iv_ref[i])
    print(f"  T={Tv:.2f}: {row}")

# ─── 2.  Train SpectralFilter ─────────────────────────────────────────────────
print("\n--- Training SpectralFilter (exact fGn PSD, H=0.1) ---")
freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
m_filter, _ = train_one(H, N, dt, psd_exact, 'SF', n_epochs=EPOCHS, device=device)

# ─── 3.  Surrogate paths ──────────────────────────────────────────────────────
print("\n--- Surrogate paths ---")
torch.manual_seed(7)
m_filter.eval()
with torch.no_grad():
    z_t     = torch.randn(N_SURR, N, device=device)
    dW_surr = m_filter(z_t).cpu().numpy()
    z_np    = z_t.cpu().numpy()

W_surr          = np.zeros((N_SURR, N + 1))
W_surr[:, 1:]   = np.cumsum(dW_surr, axis=1)
var_W           = W_surr.var(axis=0)
V_surr          = xi0 * np.exp(eta * W_surr - 0.5 * eta ** 2 * var_W[None, :])
print(f"  E[V_surr] = {V_surr.mean():.5f}  (target {xi0:.4f})")

# z-Cholesky: correlate stock BM with the filter's INPUT noise z
rng_s      = np.random.default_rng(99)
z_perp     = rng_s.standard_normal((N_SURR, N))
dZ_S_surr  = rho * z_np + np.sqrt(1.0 - rho ** 2) * z_perp   # ρ=−0.9
dZ_S_rho0  = z_perp                                            # ρ=0 baseline

# ─── 4.  Surrogate IV + delta ─────────────────────────────────────────────────
print("\n--- Computing IV surfaces ---")
iv_surr      = iv_surface(V_surr, dZ_S_surr,  dt)
iv_rho0      = iv_surface(V_surr, dZ_S_rho0,  dt)
delta_surr   = delta_surface(V_surr, dZ_S_surr,  dt)
delta_rho0   = delta_surface(V_surr, dZ_S_rho0,  dt)

# ─── 5.  RMSE summary ─────────────────────────────────────────────────────────
print("\n--- IV RMSE vs BLP reference ---")
for label, iv_s in [("SF z-Cholesky (ρ=−0.9)", iv_surr),
                    ("SF baseline   (ρ=0)    ", iv_rho0)]:
    overall = rmse_vp(iv_ref, iv_s)
    per_t   = "  ".join(
        f"{rmse_vp(iv_ref[i:i+1], iv_s[i:i+1]):.2f}"
        for i in range(len(_MATURITIES_DEFAULT)))
    print(f"  {label}  overall {overall:.2f} vp  per-T: {per_t}")

# ─── 6.  Delta summary ────────────────────────────────────────────────────────
j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
print("\n--- ATM delta (K/S₀=1.00) ---")
print(f"  {'T':>4}  {'Δ_ref':>7}  {'Δ_surr(ρ)':>10}  {'bias':>7}  "
      f"{'Δ_surr(0)':>10}  {'bias(0)':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    ref  = delta_ref[i, j_atm]
    s_rh = delta_surr[i, j_atm]
    s_r0 = delta_rho0[i, j_atm]
    print(f"  T={Tv:.2f}  {ref:7.4f}  {s_rh:10.4f}  {s_rh-ref:+7.4f}  "
          f"{s_r0:10.4f}  {s_r0-ref:+8.4f}")

# ─── 7.  Figures ──────────────────────────────────────────────────────────────
moneyness = _STRIKES_DEFAULT
fig, axes = plt.subplots(1, len(_MATURITIES_DEFAULT),
                         figsize=(5 * len(_MATURITIES_DEFAULT), 4), sharey=True)
for i, (ax, Tv) in enumerate(zip(axes, _MATURITIES_DEFAULT)):
    for iv_s, style, lbl in [
            (iv_ref,   ('k', '-',  'o'), 'BLP ref (ρ=−0.9)'),
            (iv_surr,  ('r', '--', 's'), 'SF z-Cholesky (ρ=−0.9)'),
            (iv_rho0,  ('b', ':',  '^'), 'SF baseline (ρ=0)'),
    ]:
        m = np.isfinite(iv_s[i])
        ax.plot(moneyness[m], iv_s[i][m] * 100,
                color=style[0], linestyle=style[1], marker=style[2],
                ms=4, lw=1.8 if style[1] == '-' else 1.4, label=lbl)
    ax.set_xlabel('Moneyness K/S₀')
    ax.set_title(f'T = {Tv:.2f} yr')
    if i == 0:
        ax.set_ylabel('Implied vol (%)')
    ax.legend(fontsize=7)
fig.suptitle('rBergomi ρ=−0.9: BLP reference vs SpectralFilter (z-Cholesky)', fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / 'blp_iv_comparison.png', dpi=120)
plt.close(fig)
print(f"\n  Figure → {FIGURES / 'blp_iv_comparison.png'}")

# ─── 8.  Save numerics ────────────────────────────────────────────────────────
np.savez(RESULTS / 'blp_comparison.npz',
         iv_ref=iv_ref, iv_surr=iv_surr, iv_rho0=iv_rho0,
         delta_ref=delta_ref, delta_surr=delta_surr, delta_rho0=delta_rho0,
         strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT)
print(f"  Numerics → {RESULTS / 'blp_comparison.npz'}")
print("\nDone.")
