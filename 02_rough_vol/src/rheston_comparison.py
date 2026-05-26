"""
rheston_comparison.py  —  Rough Heston: IV surface & Greeks bias at ρ=−0.9.

Model: Rough Heston  (H=0.1, κ=2, θ=0.04, ν=1.9, V₀=0.04, ρ=−0.9)

Experiments
-----------
Reference:   V_rh  + Z_rh    (ρ=−0.9)  →  IV_ref,  Δ_ref     [correct joint law]
Surrogate A: V_rh  + Z_fresh (ρ=−0.9)  →  IV_A,    Δ_A        [same V, wrong BM]
Surrogate B: V_surr+ z_surr  (ρ=−0.9)  →  IV_B,    Δ_B        [wrong V, wrong BM]
Surrogate C: V_surr+ None    (ρ=0)      →  IV_C,    Δ_C        [baseline, no leverage]

Bias A = Δ_A − Δ_ref  →  pure correlation bias (V distribution identical to reference)
Bias B = Δ_B − Δ_ref  →  correlation + V-distribution bias

Output: figures/rheston_*.png   results/rheston_comparison.npz
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance
from bochner_pinn import (SpectralFilter, get_exact_fgn_psd, train_one)
from implied_vol import (compute_iv_surface, compute_delta_surface,
                          compare_iv_surfaces,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT,
                          RESULTS, FIGURES)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
H, kappa, theta, nu, V0 = 0.10, 2.0, 0.04, 1.9, 0.04
rho   = -0.9
N, T  = 252, 1.0
dt    = T / N
N_REF = 8_000
N_SURR = 5_000
EPOCHS = 3_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Rough Heston reference paths  (V_rh, Z_rh)
# ─────────────────────────────────────────────────────────────────────────────
print("=== 1. Rough Heston simulation ===")
V_rh, Z_rh, t = simulate_rough_heston_variance(
    H, kappa, theta, nu, V0, N, N_REF, T, seed=1)
print(f"  E[V_rh] = {V_rh.mean():.5f}  (θ={theta:.4f}, bias from Euler reflection)")
print(f"  Z_rh.std() = {Z_rh.std():.4f}  (should be ≈ 1)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reference: correct joint simulation  (V_rh + Z_rh for ρ)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 2. Reference IV + delta  (V_rh, Z_rh, ρ=−0.9) ===")
iv_ref, K_arr, T_arr = compute_iv_surface(
    V_rh, dt, innovations=Z_rh, rho=rho, seed=2)
delta_ref = compute_delta_surface(V_rh, Z_rh, dt, rho=rho, seed=3)

print("  Reference IV (%):")
for i, Tv in enumerate(T_arr):
    row = "  ".join(f"{v*100:5.2f}" if np.isfinite(v) else " nan " for v in iv_ref[i])
    print(f"  T={Tv:.2f}: {row}")
print("  Reference Δ (ATM K=1.00 column):")
for i, Tv in enumerate(T_arr):
    j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
    print(f"  T={Tv:.2f}: Δ_ATM = {delta_ref[i, j_atm]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Surrogate A: same V_rh, FRESH BM for ρ  (isolates correlation bias)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. Surrogate A: V_rh + fresh Z  (pure correlation bias) ===")
rng_fresh = np.random.default_rng(seed=50)
Z_fresh   = rng_fresh.standard_normal((N_REF, N))

delta_A = compute_delta_surface(V_rh, Z_fresh, dt, rho=rho, seed=51)
bias_A  = delta_A - delta_ref

print("  Δ bias A (surrogate A − reference):")
for i, Tv in enumerate(T_arr):
    row = "  ".join(f"{b:+.4f}" for b in bias_A[i])
    print(f"  T={Tv:.2f}: {row}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Train SpectralFilter on rBergomi PSD  (fGn PSD, H=0.1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. Train SpectralFilter (exact fGn PSD, H=0.1) ===")
freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
m_filter, _ = train_one(H, N, dt, psd_exact, 'SpectralFilter [rHeston]',
                         n_epochs=EPOCHS, device=device)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Surrogate paths  (V_surr from rBergomi-trained filter)
# ─────────────────────────────────────────────────────────────────────────────
eta = nu   # reuse ν as the rBergomi η for apples-to-apples comparison
xi0 = V0

torch.manual_seed(7)
m_filter.eval()
with torch.no_grad():
    z_tensor = torch.randn(N_SURR, N, device=device)
    dW_surr  = m_filter(z_tensor).cpu().numpy()
    z_surr   = z_tensor.cpu().numpy()              # underlying white noise for ρ

W_surr = np.zeros((N_SURR, N + 1))
W_surr[:, 1:] = np.cumsum(dW_surr, axis=1)
var_W = W_surr.var(axis=0)
V_surr = xi0 * np.exp(eta * W_surr - 0.5 * eta**2 * var_W[None, :])
print(f"\n  E[V_surr] = {V_surr.mean():.5f}  (target {xi0:.4f})")
print(f"  E[V_rh]   = {V_rh.mean():.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Surrogate B: V_surr + z_surr  (wrong V distribution + wrong BM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 6. Surrogate B: V_surr + z_surr (ρ=−0.9) ===")
iv_B, _, _ = compute_iv_surface(
    V_surr, dt, innovations=z_surr, rho=rho, seed=20)
delta_B  = compute_delta_surface(V_surr, z_surr, dt, rho=rho, seed=21)
bias_B   = delta_B - delta_ref

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Surrogate C: V_surr, ρ=0  (baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 7. Surrogate C: V_surr, ρ=0  (baseline, no leverage) ===")
iv_C, _, _ = compute_iv_surface(
    V_surr, dt, innovations=None, rho=0.0, seed=30)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  IV RMSE summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 8. IV RMSE vs reference  (vol points) ===")
for label, iv_s in [
    ('Surr B  (V_surr, ρ=−0.9)', iv_B),
    ('Surr C  (V_surr, ρ=0)',     iv_C),
]:
    s = compare_iv_surfaces(iv_ref, iv_s, K_arr, T_arr)
    mats = "  ".join(f"{r:6.3f}" for r in s['rmse_by_maturity'].values())
    print(f"  {label:<30}  overall {s['rmse_overall']:6.3f}  per-T: {mats}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Delta bias summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 9. Delta bias summary ===")
j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))

print(f"\n  {'':6} {'Δ_ref':>8} {'Δ_A':>8} {'bias_A':>8} {'Δ_B':>8} {'bias_B':>8}")
for i, Tv in enumerate(T_arr):
    ref  = delta_ref[i, j_atm]
    dA   = delta_A[i, j_atm]
    dB   = delta_B[i, j_atm]
    print(f"  T={Tv:.2f} {ref:>8.4f} {dA:>8.4f} {dA-ref:>+8.4f} {dB:>8.4f} {dB-ref:>+8.4f}")

print("\n  Full bias_A table (surrogate A − reference, pure correlation bias):")
_hdr = "  " + "  ".join(f"K={k:.2f}" for k in _STRIKES_DEFAULT)
print(_hdr)
for i, Tv in enumerate(T_arr):
    row = "  ".join(f"{b:+.4f}" for b in bias_A[i])
    print(f"  T={Tv:.2f}: {row}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Figures
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 10. Saving figures ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

moneyness = K_arr / 1.0

# IV surface comparison
fig, axes = plt.subplots(1, len(T_arr), figsize=(5 * len(T_arr), 4), sharey=True)
for i, (ax, Tv) in enumerate(zip(axes, T_arr)):
    m = np.isfinite(iv_ref[i])
    ax.plot(moneyness[m], iv_ref[i][m] * 100, 'k-o', ms=4, lw=2,
            label='rHeston ref (ρ=−0.9)')
    m = np.isfinite(iv_B[i])
    ax.plot(moneyness[m], iv_B[i][m] * 100, 'r--s', ms=4, lw=1.5,
            label='Surrogate B  (ρ=−0.9)')
    m = np.isfinite(iv_C[i])
    ax.plot(moneyness[m], iv_C[i][m] * 100, 'b:^', ms=4, lw=1.5,
            label='Surrogate C  (ρ=0)')
    ax.set_xlabel('Moneyness K/S₀')
    ax.set_title(f'T = {Tv:.2f} yr')
    if i == 0:
        ax.set_ylabel('Implied vol (%)')
    ax.legend(fontsize=7)
fig.suptitle('Rough Heston: reference vs surrogate IV smiles', fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / 'rheston_iv_comparison.png', dpi=120)
plt.close(fig)

# Delta bias figure  (pure correlation bias = Surrogate A)
fig, axes = plt.subplots(1, len(T_arr), figsize=(5 * len(T_arr), 4))
for i, (ax, Tv) in enumerate(zip(axes, T_arr)):
    x = np.arange(len(K_arr))
    ax.bar(x - 0.2, bias_A[i], width=0.4, alpha=0.8,
           label='Bias A (same V, fresh BM)', color='steelblue')
    ax.bar(x + 0.2, bias_B[i], width=0.4, alpha=0.8,
           label='Bias B (surrogate V, surr BM)', color='darkorange')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k:.2f}' for k in moneyness], fontsize=7, rotation=45)
    ax.set_xlabel('Moneyness K/S₀')
    ax.set_title(f'Δ bias  T = {Tv:.2f} yr')
    if i == 0:
        ax.set_ylabel('Surrogate Δ − Reference Δ')
    ax.legend(fontsize=7)
fig.suptitle('Delta bias: leverage effect (pure ρ bias vs distribution bias)',
             fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / 'rheston_delta_bias.png', dpi=120)
plt.close(fig)

print(f"  IV comparison → {FIGURES / 'rheston_iv_comparison.png'}")
print(f"  Delta bias    → {FIGURES / 'rheston_delta_bias.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Save numerics
# ─────────────────────────────────────────────────────────────────────────────
np.savez(RESULTS / 'rheston_comparison.npz',
         K_arr=K_arr, T_arr=T_arr,
         iv_ref=iv_ref, iv_B=iv_B, iv_C=iv_C,
         delta_ref=delta_ref, delta_A=delta_A, delta_B=delta_B,
         bias_A=bias_A, bias_B=bias_B)
print(f"  Numerics      → {RESULTS / 'rheston_comparison.npz'}")
print("\n=== Done ===")
