"""
spx_calibration_iv.py  —  vega-weighted IV-direct loss + LBFGS refinement.

Two improvements over the multi-maturity calibration:

  (A) Vega-weighted price MSE.  Weight each strike by 1/vega^2 so that
      the loss approximates IV MSE to leading order:
        (dC)^2 / vega^2  ≈  (dIV)^2.
      This puts equal emphasis on OTM and ATM strikes (vega is small at
      OTM, so 1/vega^2 is large) — better than uniform price MSE.

  (B) LBFGS final refinement.  Adam handles the wide parts of the loss
      surface but oscillates near the minimum.  After 400 Adam epochs we
      switch to LBFGS (full-batch, second-order curvature) for 30 iters.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import time
from datetime import datetime
from pathlib import Path
from scipy.stats import norm

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from power_volterra import PowerVolterraSurrogate
from implied_vol import implied_vol_single, RESULTS

# ─── 1.  SPY multi-maturity chain (same as spx_calibration_multi.py) ─────────
print("--- Fetching SPY multi-maturity chain ---")
spy   = yf.Ticker("SPY")
S0    = float(spy.fast_info['lastPrice'])
exps  = spy.options
today = datetime.utcnow().date()
def days_to(e): return (datetime.strptime(e, "%Y-%m-%d").date() - today).days

target_days_list = [30, 90, 180]
chosen_exps = []
for td in target_days_list:
    cand = min(exps, key=lambda e: abs(days_to(e) - td))
    if cand not in chosen_exps:
        chosen_exps.append(cand)
print(f"  Spot SPY: ${S0:.2f}, expirations: {chosen_exps}")

T_years_list = [days_to(e) / 365.25 for e in chosen_exps]

all_K, all_C, all_IVmkt, all_T = [], [], [], []
strikes_target_ratio = np.array([0.85, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15])
for exp_str, T_yr in zip(chosen_exps, T_years_list):
    oc = spy.option_chain(exp_str)
    calls = oc.calls
    calls = calls[(calls['strike'] >= 0.80 * S0) & (calls['strike'] <= 1.20 * S0)]
    calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
    calls['mid']    = 0.5 * (calls['bid'] + calls['ask'])
    calls['spread'] = (calls['ask'] - calls['bid']) / calls['mid']
    calls = calls[calls['spread'] < 0.20].sort_values('strike').reset_index(drop=True)
    chosen_idx = []
    for K_t in strikes_target_ratio * S0:
        j = (calls['strike'] - K_t).abs().idxmin()
        if j not in chosen_idx:
            chosen_idx.append(j)
    sel = calls.loc[chosen_idx]
    for _, row in sel.iterrows():
        all_K.append(row['strike'])
        all_C.append(row['mid'])
        all_IVmkt.append(row['impliedVolatility'])
        all_T.append(T_yr)
all_K, all_C, all_IVmkt, all_T = map(np.array, (all_K, all_C, all_IVmkt, all_T))
N_OPT = len(all_K)
K_norm, C_norm = all_K / S0, all_C / S0
print(f"  Total calibration points: {N_OPT}")


# ─── 2.  Pre-compute vega weights from market IVs ────────────────────────────
# vega = S * phi(d1) * sqrt(T),  d1 = (ln(S/K) + 0.5*IV^2*T) / (IV*sqrt(T))
d1_mkt = (np.log(S0 / all_K) + 0.5 * all_IVmkt ** 2 * all_T) / (all_IVmkt * np.sqrt(all_T))
vega_mkt = S0 * norm.pdf(d1_mkt) * np.sqrt(all_T)
# Normalised to S0=1 convention
vega_norm = vega_mkt / S0
# Weights = 1/vega^2 (clamped to avoid blow-up at deepest OTM)
weight = 1.0 / np.maximum(vega_norm, 0.02) ** 2
weight = weight / weight.mean()      # normalise mean weight to 1
print(f"  Vega range: [{vega_norm.min():.4f}, {vega_norm.max():.4f}]")
print(f"  Weight range: [{weight.min():.2f}, {weight.max():.2f}]")


# ─── 3.  Surrogate ────────────────────────────────────────────────────────────
H, RHO   = 0.10, -0.85
N        = 252
T_MAX    = max(T_years_list)
DT       = T_MAX / N

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Surrogate: ρ={RHO}, T_max={T_MAX:.4f}y, N={N}")

short_T = min(T_years_list)
ATM_short = next(iv for K, T_i, iv in zip(all_K, all_T, all_IVmkt)
                  if abs(T_i - short_T) < 1e-6 and abs(K - S0) < 5)
V_PRIOR  = float(ATM_short ** 2)

model = PowerVolterraSurrogate(
    N=N, dt=DT, H=H, V_0=V_PRIOR,
    theta_0=V_PRIOR, kappa_0=2.0, nu_0=2.0,
    beta_0=0.5, V_min=1e-6).to(device)

T_idx_arr = torch.tensor(
    [min(int(round(T_i / DT)), N) for T_i in all_T], device=device)
K_arr_t  = torch.tensor(K_norm, dtype=torch.float32, device=device)
C_t      = torch.tensor(C_norm, dtype=torch.float32, device=device)
W_t      = torch.tensor(weight, dtype=torch.float32, device=device)

print(f"  Init: θ={model.theta.item():.4f}, c_s={model.c_stoch.item():.3e}, "
      f"β={model.beta.item():.3f}")


# ─── 4.  Common-random-number noise ──────────────────────────────────────────
N_TRAIN = 128
torch.manual_seed(42)
z_train  = torch.randn(N_TRAIN, N, device=device)
z_perp_t = torch.randn(N_TRAIN, N, device=device)


def surr_prices(model):
    V    = model(z_train)
    V_l  = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z_train + np.sqrt(1 - RHO ** 2) * z_perp_t
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S   = torch.cat([torch.zeros(V.shape[0], 1, device=device),
                          log_inc.cumsum(dim=1)], dim=1)
    S = torch.exp(log_S)
    C = torch.zeros(N_OPT, device=device)
    for i in range(N_OPT):
        C[i] = (S[:, T_idx_arr[i]] - K_arr_t[i]).clamp(min=0.0).mean()
    return C


def vega_weighted_loss(C_surr):
    return (W_t * (C_surr - C_t) ** 2).mean()


# ─── 5.  Phase 1: Adam with vega-weighted loss ───────────────────────────────
print("\n--- Phase 1: Adam, vega-weighted MSE (400 epochs) ---")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_L, best_state, best_ep = float('inf'), None, 0
t0 = time.time()
for ep in range(400):
    model.train()
    L = vega_weighted_loss(surr_prices(model))
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if ep >= 30 and L.item() < best_L:
        best_L = L.item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_ep = ep
    if ep % 50 == 0 or ep == 399:
        print(f"  ep {ep:4d}  L_w {L.item():.2e}  θ={model.theta.item():.4f}  "
              f"β={model.beta.item():.3f}")
print(f"  Best L_w = {best_L:.2e} at ep {best_ep}; loading best state")
model.load_state_dict(best_state)
print(f"  Adam time: {time.time() - t0:.1f} s")


# ─── 6.  Phase 2: LBFGS refinement ────────────────────────────────────────────
print("\n--- Phase 2: LBFGS refinement (30 iters) ---")
lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=30,
                          history_size=20, line_search_fn='strong_wolfe')

def closure():
    lbfgs.zero_grad()
    L = vega_weighted_loss(surr_prices(model))
    L.backward()
    return L

t1 = time.time()
L_final = lbfgs.step(closure)
print(f"  LBFGS time: {time.time() - t1:.1f} s")
print(f"  Post-LBFGS L_w: {L_final.item():.2e}")
print(f"  Final  θ={model.theta.item():.4f}  c_d={model.c_drift.item():.3e}  "
      f"c_s={model.c_stoch.item():.3e}  β={model.beta.item():.3f}")


# ─── 7.  Evaluate ─────────────────────────────────────────────────────────────
print("\n--- Evaluating fit on full surface (5k MC eval paths) ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_eval = torch.randn(5_000, N, device=device)
    z_perp_eval = torch.randn(5_000, N, device=device)
    V = model(z_eval)
    V_l = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z_eval + np.sqrt(1 - RHO ** 2) * z_perp_eval
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S = torch.cat([torch.zeros(V.shape[0], 1, device=device),
                       log_inc.cumsum(dim=1)], dim=1)
    S_eval = torch.exp(log_S).cpu().numpy()

C_surr = np.array([
    np.mean(np.maximum(S_eval[:, int(T_idx_arr[i].item())] - K_norm[i], 0.0))
    for i in range(N_OPT)
])
C_surr_usd = C_surr * S0
IV_surr   = np.array([implied_vol_single(c, S0, K, 0.0, T_i)
                       for c, K, T_i in zip(C_surr_usd, all_K, all_T)])
IV_market = np.array([implied_vol_single(c, S0, K, 0.0, T_i)
                       for c, K, T_i in zip(all_C, all_K, all_T)])

print("\n" + "=" * 92)
print(f"    SPY MULTI-MATURITY CALIBRATION (vega-weighted + LBFGS, S0=${S0:.2f})")
print("=" * 92)
iv_diffs_all = []
for exp_str, T_i in zip(chosen_exps, T_years_list):
    mask = np.isclose(all_T, T_i)
    print(f"\n  --- {exp_str}  (T = {T_i:.4f} yr, {days_to(exp_str)} days) ---")
    diffs_T = []
    for K, iv_m, iv_s in zip(all_K[mask], IV_market[mask], IV_surr[mask]):
        if np.isfinite(iv_m) and np.isfinite(iv_s):
            d = (iv_s - iv_m) * 100
            diffs_T.append(d)
            iv_diffs_all.append(d)
            print(f"  ${K:6.2f} K/S0={K/S0:.3f}  IV_mkt={iv_m*100:6.2f}%  "
                  f"IV_surr={iv_s*100:6.2f}%  err {d:+6.2f} vp")
    if diffs_T:
        print(f"  → maturity IV RMSE: {np.sqrt(np.mean(np.array(diffs_T)**2)):.2f} vp")

iv_diffs_all = np.array(iv_diffs_all)
print("\n" + "=" * 92)
print(f"  OVERALL IV RMSE: {np.sqrt(np.mean(iv_diffs_all**2)):.2f} vp"
      f"  (mean abs: {np.mean(np.abs(iv_diffs_all)):.2f} vp)"
      f"  (N = {len(iv_diffs_all)})")
print("=" * 92)

np.savez(RESULTS / 'spx_calibration_iv.npz',
         S0=S0, all_T=all_T, all_K=all_K,
         IV_market=IV_market, IV_surr=IV_surr,
         theta_final=model.theta.item(), beta_final=model.beta.item(),
         c_drift_final=model.c_drift.item(), c_stoch_final=model.c_stoch.item())
print(f"\nSaved → {RESULTS / 'spx_calibration_iv.npz'}")
