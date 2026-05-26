"""
spx_calibration.py  —  calibrate PowerVolterraSurrogate to live SPY option chain.

Pulls SPY mid prices from Yahoo Finance, picks a mid-term expiration,
filters to liquid OTM puts + OTM calls spanning the smile, and trains
the joint Volterra surrogate (kernel + scalars + β) to match the call
prices.  ρ fixed at −0.7 (typical SPX leverage).

This is the real-market analog of the synthetic rough Heston experiments
in §5.4.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import time
from datetime import datetime
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from power_volterra import PowerVolterraSurrogate
from implied_vol import (implied_vol_single, RESULTS)

# ─── 1.  Pull SPY option chain ────────────────────────────────────────────────
print("--- Fetching SPY option chain ---")
spy   = yf.Ticker("SPY")
S0    = float(spy.fast_info['lastPrice'])
exps  = spy.options
print(f"  Spot SPY: ${S0:.2f}")
print(f"  Available expirations (first 10): {exps[:10]}")

# Pick the expiration closest to ~120 days out (mid-term)
today = datetime.utcnow().date()
def days_to(exp_str):
    return (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days

# Find expiration closest to 120 days
days_list = [(e, days_to(e)) for e in exps]
target_days = 120
chosen = min(days_list, key=lambda x: abs(x[1] - target_days))
chosen_exp, T_days = chosen
T_years = T_days / 365.25
print(f"  Chosen expiration: {chosen_exp}  ({T_days} days → T = {T_years:.4f} yr)")

oc = spy.option_chain(chosen_exp)
calls = oc.calls

# Filter: K/S0 in [0.80, 1.20], non-zero bid+ask, narrow spread
calls = calls[(calls['strike'] >= 0.80 * S0) & (calls['strike'] <= 1.20 * S0)]
calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
calls['mid']     = 0.5 * (calls['bid'] + calls['ask'])
calls['spread']  = (calls['ask'] - calls['bid']) / calls['mid']
calls = calls[calls['spread'] < 0.20]      # spread under 20% of mid
calls = calls.sort_values('strike').reset_index(drop=True)
print(f"  {len(calls)} liquid options after filtering")
print(calls[['strike', 'bid', 'ask', 'mid', 'impliedVolatility']].head(12).to_string())

# Pick a tractable subset of ~9 strikes spanning K/S0 ∈ [0.85, 1.15]
strikes_target_ratio = np.array([0.85, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15])
K_targets = strikes_target_ratio * S0

# For each target K, pick the nearest available strike
chosen_idx = []
for K_t in K_targets:
    j = (calls['strike'] - K_t).abs().idxmin()
    if j not in chosen_idx:
        chosen_idx.append(j)
calls_sel = calls.loc[chosen_idx].reset_index(drop=True)
K_market  = calls_sel['strike'].values
C_market  = calls_sel['mid'].values
IV_market = calls_sel['impliedVolatility'].values
print(f"\n  Selected {len(calls_sel)} strikes for calibration:")
for K, C, iv in zip(K_market, C_market, IV_market):
    print(f"    K = ${K:6.2f}  (K/S0={K/S0:.3f})  C_mid = ${C:6.2f}  IV(yf) = {iv*100:.2f}%")

# Normalise to S0 = 1 (our surrogate convention)
K_norm = K_market / S0
C_norm = C_market / S0


# ─── 2.  Surrogate setup ──────────────────────────────────────────────────────
H, V0   = 0.10, 0.030         # rough vol H≈0.1; SPY VIX-implied vol ≈ 17% so V0≈0.03
RHO     = -0.85               # SPX-typical leverage (steep skew)
N       = 252
T       = T_years             # match market expiration
DT      = T / N

# ATM-IV-implied prior on E[V_T] for the moment regulariser
# (Black-Scholes-equivalent: ATM IV^2 ≈ E[V_T])
atm_iv_idx = int(np.argmin(np.abs(K_market - S0)))
ATM_IV     = IV_market[atm_iv_idx]
V_PRIOR    = float(ATM_IV ** 2)
print(f"  ATM IV = {ATM_IV*100:.2f}%  →  V_T prior ≈ {V_PRIOR:.4f}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Surrogate config: H={H}, V_0={V0}, ρ={RHO}, T={T:.4f}y, N={N}, dt={DT:.5f}")

# Start from market-prior guess for SPX:
# - V_0 = theta_0 = ATM IV^2 (long-run vol matches near-term ATM)
# - nu_0 = 2.0  (typical SPX vol-of-vol)
THETA_INIT = V_PRIOR
V0         = V_PRIOR
model = PowerVolterraSurrogate(
    N=N, dt=DT, H=H, V_0=V0,
    theta_0=THETA_INIT, kappa_0=2.0, nu_0=2.0,
    beta_0=0.5, V_min=1e-6).to(device)
print(f"  Init: θ={model.theta.item():.4f}, c_d={model.c_drift.item():.3e}, "
      f"c_s={model.c_stoch.item():.3e}, β={model.beta.item():.3f}")


# ─── 3.  Training: minimise call-price MSE (only this maturity) ──────────────
T_idx = N    # the very last column of V is at time T (= market expiry)

K_arr = torch.tensor(K_norm, dtype=torch.float32, device=device)
C_t   = torch.tensor(C_norm, dtype=torch.float32, device=device)


def surr_prices(model, z, z_perp):
    V    = model(z)
    V_l  = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z + np.sqrt(1 - RHO ** 2) * z_perp
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S   = torch.cat([torch.zeros(V.shape[0], 1, device=device),
                          log_inc.cumsum(dim=1)], dim=1)
    S = torch.exp(log_S)
    S_T = S[:, T_idx]
    return torch.stack([(S_T - K).clamp(min=0.0).mean() for K in K_arr])


print("\n--- Training to match SPY mid prices (Adam + best-checkpoint, no premature stop) ---")
N_TRAIN, EPOCHS, LR = 256, 500, 1e-3
torch.manual_seed(42)
z_train  = torch.randn(N_TRAIN, N, device=device)
z_perp_t = torch.randn(N_TRAIN, N, device=device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
best_L, best_state, best_ep = float('inf'), None, 0
losses = []
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    V_s = model(z_train)
    V_l = V_s[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z_train + np.sqrt(1 - RHO**2) * z_perp_t
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S = torch.cat([torch.zeros(V_s.shape[0], 1, device=device),
                        log_inc.cumsum(dim=1)], dim=1)
    S = torch.exp(log_S)
    C_s = torch.stack([(S[:, N] - K).clamp(min=0.0).mean() for K in K_arr])
    L = F.mse_loss(C_s, C_t)
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(L.item())
    # Snapshot best after at least 30 warmup epochs (avoid trivial 'init is best')
    if ep >= 30 and L.item() < best_L:
        best_L = L.item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_ep = ep
    if ep % 50 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep:4d}  L {L.item():.2e}  "
              f"θ={model.theta.item():.4f}  β={model.beta.item():.3f}  "
              f"E[V_T]={V_s[:,-1].mean().item():.4f}")
if best_state is not None:
    print(f"  Best L = {best_L:.2e} at ep {best_ep}; restoring")
    model.load_state_dict(best_state)
print(f"  Training time: {time.time() - t0:.1f} s")
print(f"  Final  θ={model.theta.item():.4f}  c_d={model.c_drift.item():.3e}  "
      f"c_s={model.c_stoch.item():.3e}  β={model.beta.item():.3f}")


# ─── 4.  Evaluate: compute surrogate IV smile and compare to market ─────────
print("\n--- Evaluating fit ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_eval  = torch.randn(5_000, N, device=device)
    z_perp  = torch.randn(5_000, N, device=device)
    C_surr  = surr_prices(model, z_eval, z_perp).cpu().numpy()

# Convert back to dollar prices and to IVs
C_surr_usd = C_surr * S0
IV_surr = np.array([
    implied_vol_single(c, S0, K, 0.0, T)
    for c, K in zip(C_surr_usd, K_market)
])
IV_market_recomputed = np.array([
    implied_vol_single(c, S0, K, 0.0, T)
    for c, K in zip(C_market, K_market)
])

print("\n" + "=" * 84)
print(f"    SPY OPTION CHAIN CALIBRATION  (exp {chosen_exp}, T={T_years:.4f} yr, "
      f"S0=${S0:.2f})")
print("=" * 84)
print(f"  {'K':>8}  {'K/S0':>5}  {'C_mkt':>8}  {'C_surr':>8}  "
      f"{'IV_mkt':>7}  {'IV_surr':>7}  {'IV err (vp)':>11}")
print("-" * 84)
iv_diffs = []
for K, c_m, c_s_v, iv_m, iv_s_v in zip(
        K_market, C_market, C_surr_usd, IV_market_recomputed, IV_surr):
    if np.isfinite(iv_m) and np.isfinite(iv_s_v):
        d = (iv_s_v - iv_m) * 100
        iv_diffs.append(d)
    else:
        d = np.nan
    print(f"  ${K:6.2f}  {K/S0:.3f}  ${c_m:6.2f}  ${c_s_v:6.2f}  "
          f"{iv_m*100:6.2f}%  {iv_s_v*100:6.2f}%  {d:+9.2f}")
print("=" * 84)
iv_diffs = np.array(iv_diffs)
print(f"  IV calibration RMSE: {np.sqrt(np.mean(iv_diffs**2)):.2f} vp"
      f"  (mean abs: {np.mean(np.abs(iv_diffs)):.2f} vp)")
print(f"  Bid/ask spread implies IV uncertainty of ~ "
      f"{0.5 * np.mean([s for s in calls_sel['spread']]) * 100:.2f}%")

np.savez(RESULTS / 'spx_calibration.npz',
         S0=S0, T=T, chosen_exp=chosen_exp,
         K_market=K_market, C_market=C_market, IV_market=IV_market_recomputed,
         C_surr=C_surr_usd, IV_surr=IV_surr,
         theta_final=model.theta.item(), beta_final=model.beta.item(),
         c_drift_final=model.c_drift.item(), c_stoch_final=model.c_stoch.item())
print(f"\nSaved → {RESULTS / 'spx_calibration.npz'}")
