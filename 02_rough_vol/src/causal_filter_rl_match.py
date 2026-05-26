"""
causal_filter_rl_match.py  —  closing the residual −0.07 delta bias.

Hypothesis (from §5.4): the residual ATM delta bias of −0.07 comes from
the V-distribution gap between the RL-discretised BLP reference and the
Wood--Chan-based fGn PSD used in training.  If true, training the causal
filter on the *RL-fBm* PSD (matched to the reference) should eliminate
that residual.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_blp, compute_psd
from bochner_pinn import (CausalSpectralFilter, get_exact_fgn_psd, train_one)
from implied_vol import (implied_vol_single,
                         _STRIKES_DEFAULT, _MATURITIES_DEFAULT, RESULTS, FIGURES)
from scipy.special import gamma as _gamma
from scipy.linalg import toeplitz as _toeplitz

# ─── Parameters ───────────────────────────────────────────────────────────────
H, ETA, XI0 = 0.10, 1.9, 0.04
RHO          = -0.9
N, T         = 252, 1.0
DT           = T / N
N_REF        = 8_000
N_SURR       = 5_000
EPOCHS       = 3_000
N_MC_PSD     = 100_000   # for estimating the RL-fBm PSD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  RL-fBm PSD estimation  ───────────────────────────────────────────────
def compute_rl_fgn_psd(H, N, dt, n_mc=N_MC_PSD, seed=11):
    """
    Monte Carlo PSD of the RL-discretised fGn increments matching
    simulate_rbm_blp's variance path construction.
    """
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((n_mc, N))
    l   = np.arange(1, N + 1, dtype=np.float64)
    g   = l ** (H - 0.5) / _gamma(H + 0.5)
    M   = np.tril(_toeplitz(g, np.r_[g[0], np.zeros(N - 1)]))
    W_H = np.zeros((n_mc, N + 1))
    W_H[:, 1:] = dt ** H * (Z @ M.T)
    dW_H = np.diff(W_H, axis=1)        # (n_mc, N) RL-fGn increments
    _, psd = compute_psd(dW_H, dt)
    return psd


print("--- Computing PSDs ---")
freqs, psd_woodchan = get_exact_fgn_psd(H, N, DT)
print("  Computing RL-fBm increments PSD... ", end='', flush=True)
psd_rl = compute_rl_fgn_psd(H, N, DT)
print("done.")
print(f"  Wood-Chan PSD: low={psd_woodchan[:5].mean():.3e}  high={psd_woodchan[-5:].mean():.3e}")
print(f"  RL-fBm   PSD: low={psd_rl[:5].mean():.3e}  high={psd_rl[-5:].mean():.3e}")
ratio = psd_rl / psd_woodchan
print(f"  Ratio psd_rl/psd_wc: min={ratio.min():.3f}  max={ratio.max():.3f}  median={np.median(ratio):.3f}")


# ─── 2.  Train two causal filters: Wood-Chan PSD vs RL PSD  ──────────────────
print("\n--- Training causal filters ---")
model_wc = CausalSpectralFilter(N, H=H, dt=DT)
model_wc, _ = train_one(H, N, DT, psd_woodchan, 'Causal[WoodChan]',
                        n_epochs=EPOCHS, device=device, model=model_wc)

model_rl = CausalSpectralFilter(N, H=H, dt=DT)
model_rl, _ = train_one(H, N, DT, psd_rl, 'Causal[RL-fBm]',
                        n_epochs=EPOCHS, device=device, model=model_rl)


# ─── 3.  BLP reference  ───────────────────────────────────────────────────────
print("\n--- BLP reference (ρ=−0.9) ---")
V_ref, dZ_S_ref, _ = simulate_rbm_blp(H, ETA, XI0, N, N_REF, T, rho=RHO, seed=1)


# ─── 4.  IV + delta helpers ────────────────────────────────────────────────
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
            idx  = min(int(round(Tv / dt)), V.shape[1] - 1)
            p[i] = np.mean(np.maximum(S[:, idx] - K_atm, 0.0))
        return p

    return (_price(1.0 + bump) - _price(1.0 - bump)) / (2.0 * bump)


def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan


def surr_pack(model, n, seed_z=7, seed_perp=99):
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
    return V, dZ_rho


# ─── 5.  Evaluate both filters ───────────────────────────────────────────────
print("\n--- Evaluating surrogates ---")
iv_ref       = iv_blp(V_ref, dZ_S_ref, DT)
delta_ref    = delta_atm_blp(V_ref, dZ_S_ref, DT)

V_wc, dZ_wc  = surr_pack(model_wc, N_SURR)
V_rl, dZ_rl  = surr_pack(model_rl, N_SURR)

print(f"  E[V_wc] = {V_wc.mean():.5f}    E[V_rl] = {V_rl.mean():.5f}    target {XI0}")

iv_wc        = iv_blp(V_wc, dZ_wc, DT)
iv_rl        = iv_blp(V_rl, dZ_rl, DT)
delta_wc     = delta_atm_blp(V_wc, dZ_wc, DT)
delta_rl     = delta_atm_blp(V_rl, dZ_rl, DT)

# ─── 6.  Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("        CLOSING THE RESIDUAL: train PSD = Wood-Chan vs RL-fBm")
print("=" * 78)
print(f"  {'Training PSD':<14} | {'IV RMSE':>9} | {'ATM Δ T=.25':>11} | {'ATM Δ T=.50':>11} | {'ATM Δ T=1.0':>11}")
print(f"  {'(reference)':<14} | {'  —':>9} | {delta_ref[0]:>11.4f} | {delta_ref[1]:>11.4f} | {delta_ref[2]:>11.4f}")
print("-" * 78)
for label, iv_s, d in [
    ('Wood-Chan',  iv_wc, delta_wc),
    ('RL-fBm',     iv_rl, delta_rl),
]:
    rmse = rmse_vp(iv_ref, iv_s)
    bias = d - delta_ref
    print(f"  {label:<14} | {rmse:>9.2f} | {d[0]:>5.4f}({bias[0]:+5.3f}) | "
          f"{d[1]:>5.4f}({bias[1]:+5.3f}) | {d[2]:>5.4f}({bias[2]:+5.3f})")
print("=" * 78)

np.savez(RESULTS / 'causal_filter_rl_match.npz',
         iv_ref=iv_ref, iv_wc=iv_wc, iv_rl=iv_rl,
         delta_ref=delta_ref, delta_wc=delta_wc, delta_rl=delta_rl,
         psd_woodchan=psd_woodchan, psd_rl=psd_rl)
print(f"\nSaved → {RESULTS / 'causal_filter_rl_match.npz'}")
