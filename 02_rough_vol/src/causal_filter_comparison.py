"""
causal_filter_comparison.py  —  CausalSpectralFilter + z-Cholesky for ρ=−0.9.

Hypothesis: replacing the circular FFT with a causal convolution eliminates
the wrap-around pathology, making z-Cholesky a valid ρ-correlation scheme.

Setup
-----
  Reference  : BLP rBergomi (RL discretisation, ρ=−0.9)
  Causal SF  : CausalSpectralFilter trained on exact fGn PSD, z-Cholesky ρ=−0.9
  Circular SF: SpectralFilter (circular FFT), z-Cholesky ρ=−0.9  [negative control]
  Baseline   : CausalSpectralFilter, ρ=0

Key question: does the causal filter cut the 204-vp failure of the circular filter?
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_blp
from bochner_pinn import (SpectralFilter, CausalSpectralFilter,
                           get_exact_fgn_psd, train_one, bochner_loss)
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

def iv_surface_blp(V, dZ_S, dt, S0=1.0, r=0.0,
                   strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT):
    """IV surface from variance V and pre-correlated BM innovations dZ_S."""
    V_left  = V[:, :-1].clip(0)
    log_inc = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ_S
    n       = V.shape[0]
    log_S   = np.log(S0) + np.concatenate(
        [np.zeros((n, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    S = np.exp(log_S)
    K_arr = strikes * S0
    iv = np.full((len(maturities), len(strikes)), np.nan)
    for i, Tv in enumerate(maturities):
        idx = min(int(round(Tv / dt)), V.shape[1] - 1)
        S_T = S[:, idx]
        for j, K in enumerate(K_arr):
            price    = np.mean(np.maximum(S_T - K, 0.0)) * np.exp(-r * Tv)
            iv[i, j] = implied_vol_single(price, S0, K, r, Tv)
    return iv


def delta_blp(V, dZ_S, dt, S0=1.0, r=0.0, bump=0.01,
              strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT):
    """FD delta with dZ_S frozen across S0 bumps."""
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
    return np.sqrt(np.mean(d[m] ** 2)) if m.any() else np.nan


def surr_paths(model, n, seed_z=7, seed_perp=99):
    """Generate (V_surr, dZ_S_rho, dZ_S_rho0) from a filter model."""
    torch.manual_seed(seed_z)
    model.eval()
    with torch.no_grad():
        z_t  = torch.randn(n, N, device=device)
        dW   = model(z_t).cpu().numpy()
        z_np = z_t.cpu().numpy()
    W          = np.zeros((n, N + 1))
    W[:, 1:]   = np.cumsum(dW, axis=1)
    var_W      = W.var(axis=0)
    V          = xi0 * np.exp(eta * W - 0.5 * eta ** 2 * var_W[None, :])
    rng        = np.random.default_rng(seed_perp)
    z_perp     = rng.standard_normal((n, N))
    dZ_rho     = rho * z_np + np.sqrt(1.0 - rho ** 2) * z_perp
    dZ_rho0    = z_perp
    return V, dZ_rho, dZ_rho0


# ─── 1. BLP reference ─────────────────────────────────────────────────────────
print("--- BLP reference (ρ=−0.9) ---")
V_ref, dZ_S_ref, _ = simulate_rbm_blp(H, eta, xi0, N, N_REF, T, rho=rho, seed=1)
print(f"  E[V_ref] = {V_ref.mean():.5f}  (target {xi0:.4f})")
iv_ref    = iv_surface_blp(V_ref, dZ_S_ref, dt)
delta_ref = delta_blp(V_ref, dZ_S_ref, dt)
print("  Reference IV (%):")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    row = "  ".join(f"{v*100:5.1f}" if np.isfinite(v) else "  nan" for v in iv_ref[i])
    print(f"  T={Tv:.2f}: {row}")

# ─── 2. Train both filters on exact fGn PSD ───────────────────────────────────
print("\n--- Training filters ---")
freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
psd_t = torch.tensor(psd_exact, dtype=torch.float32, device=device)

# Causal SpectralFilter
causal_model = CausalSpectralFilter(N, H=H, dt=dt)
causal_model, _ = train_one(H, N, dt, psd_exact, 'CausalSF', n_epochs=EPOCHS,
                             device=device, model=causal_model)

# Circular SpectralFilter (negative control)
circ_model, _ = train_one(H, N, dt, psd_exact, 'CircularSF', n_epochs=EPOCHS,
                           device=device)

# ─── 3. Surrogate paths & IV surfaces ────────────────────────────────────────
print("\n--- Evaluating surrogates ---")

V_c, dZ_c_rho, dZ_c_rho0 = surr_paths(causal_model,  N_SURR, seed_z=7, seed_perp=99)
V_f, dZ_f_rho, dZ_f_rho0 = surr_paths(circ_model,    N_SURR, seed_z=7, seed_perp=99)

print(f"  E[V_causal]  = {V_c.mean():.5f}")
print(f"  E[V_circ]    = {V_f.mean():.5f}")

iv_c_rho  = iv_surface_blp(V_c, dZ_c_rho,  dt)
iv_c_rho0 = iv_surface_blp(V_c, dZ_c_rho0, dt)
iv_f_rho  = iv_surface_blp(V_f, dZ_f_rho,  dt)

delta_c_rho  = delta_blp(V_c, dZ_c_rho,  dt)
delta_c_rho0 = delta_blp(V_c, dZ_c_rho0, dt)

# ─── 4. RMSE summary ──────────────────────────────────────────────────────────
print("\n--- IV RMSE vs BLP reference ---")
rows = [
    ("Causal SF   z-Cholesky (ρ=−0.9)", iv_c_rho),
    ("Causal SF   baseline   (ρ=0)   ", iv_c_rho0),
    ("Circular SF z-Cholesky (ρ=−0.9)", iv_f_rho),
]
for label, iv_s in rows:
    overall = rmse_vp(iv_ref, iv_s)
    per_t   = "  ".join(
        f"{rmse_vp(iv_ref[i:i+1], iv_s[i:i+1]):.2f}"
        for i in range(len(_MATURITIES_DEFAULT)))
    print(f"  {label}  overall {overall:7.2f} vp  per-T: {per_t}")

# ─── 5. Delta summary ─────────────────────────────────────────────────────────
j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
print("\n--- ATM delta (K/S₀=1.00) ---")
print(f"  {'T':>4}  {'Δ_ref':>7}  {'Δ_caus(ρ)':>10}  {'bias':>7}  "
      f"{'Δ_caus(0)':>10}  {'bias(0)':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    ref = delta_ref[i, j_atm]
    crh = delta_c_rho[i, j_atm]
    cr0 = delta_c_rho0[i, j_atm]
    print(f"  T={Tv:.2f}  {ref:7.4f}  {crh:10.4f}  {crh-ref:+7.4f}  "
          f"{cr0:10.4f}  {cr0-ref:+8.4f}")

# ─── 6. Figure ────────────────────────────────────────────────────────────────
moneyness = _STRIKES_DEFAULT
fig, axes = plt.subplots(1, len(_MATURITIES_DEFAULT),
                         figsize=(5 * len(_MATURITIES_DEFAULT), 4), sharey=True)
for i, (ax, Tv) in enumerate(zip(axes, _MATURITIES_DEFAULT)):
    for iv_s, col, ls, mk, lbl in [
        (iv_ref,    'k', '-',  'o', 'BLP ref (ρ=−0.9)'),
        (iv_c_rho,  'g', '--', 's', 'Causal SF z-Chol (ρ=−0.9)'),
        (iv_c_rho0, 'b', ':',  '^', 'Causal SF (ρ=0)'),
        (iv_f_rho,  'r', '-.', 'x', 'Circular SF z-Chol (ρ=−0.9)'),
    ]:
        m = np.isfinite(iv_s[i])
        ax.plot(moneyness[m], iv_s[i][m] * 100,
                color=col, linestyle=ls, marker=mk, ms=4,
                lw=1.8 if ls == '-' else 1.3, label=lbl)
    ax.set_xlabel('Moneyness K/S₀')
    ax.set_title(f'T = {Tv:.2f} yr')
    if i == 0:
        ax.set_ylabel('Implied vol (%)')
    ax.legend(fontsize=7)
fig.suptitle('Causal vs Circular SpectralFilter: ρ=−0.9 z-Cholesky', fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / 'causal_filter_comparison.png', dpi=120)
plt.close(fig)
print(f"\n  Figure → {FIGURES / 'causal_filter_comparison.png'}")

# ─── 7. Save ──────────────────────────────────────────────────────────────────
np.savez(RESULTS / 'causal_filter_comparison.npz',
         iv_ref=iv_ref, iv_c_rho=iv_c_rho, iv_c_rho0=iv_c_rho0, iv_f_rho=iv_f_rho,
         delta_ref=delta_ref, delta_c_rho=delta_c_rho, delta_c_rho0=delta_c_rho0,
         strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT)
print(f"  Numerics → {RESULTS / 'causal_filter_comparison.npz'}")
print("\nDone.")
