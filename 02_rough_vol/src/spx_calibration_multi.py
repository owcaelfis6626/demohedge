"""
spx_calibration_multi.py  —  multi-maturity SPY calibration.

Single-maturity calibration (§5.4) leaves the term structure of the
kernel under-constrained.  Here we calibrate to THREE expirations
simultaneously (~30, ~90, ~180 days), sharing all parameters (theta,
c_drift, c_stoch, beta, w_i) across maturities.  The Volterra surrogate
generates a single 1-year path; call prices at each expiry T_i read
from V[:, idx(T_i)].
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
from implied_vol import implied_vol_single, RESULTS

# ─── 1.  SPY chain at multiple expirations ────────────────────────────────────
print("--- Fetching SPY multi-maturity option chain ---")
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
print(f"  Spot SPY: ${S0:.2f}")
print(f"  Chosen expirations: {chosen_exps}")
T_days_list  = [days_to(e) for e in chosen_exps]
T_years_list = [d / 365.25 for d in T_days_list]
print(f"  T (years): {[f'{t:.4f}' for t in T_years_list]}")

# Collect (K, C_mid, IV_yf, T) tuples across all maturities
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
    K_targets = strikes_target_ratio * S0
    for K_t in K_targets:
        j = (calls['strike'] - K_t).abs().idxmin()
        if j not in chosen_idx:
            chosen_idx.append(j)
    sel = calls.loc[chosen_idx]
    for _, row in sel.iterrows():
        all_K.append(row['strike'])
        all_C.append(row['mid'])
        all_IVmkt.append(row['impliedVolatility'])
        all_T.append(T_yr)
    print(f"  {exp_str} ({days_to(exp_str)}d): {len(sel)} strikes  "
          f"K range [{sel['strike'].min():.0f}, {sel['strike'].max():.0f}]  "
          f"ATM IV {sel.iloc[len(sel)//2]['impliedVolatility']*100:.2f}%")

all_K     = np.array(all_K)
all_C     = np.array(all_C)
all_IVmkt = np.array(all_IVmkt)
all_T     = np.array(all_T)
N_OPT     = len(all_K)
print(f"\n  Total calibration points: {N_OPT}")

# Normalise K to S0=1 convention
K_norm  = all_K / S0
C_norm  = all_C / S0


# ─── 2.  Surrogate setup ──────────────────────────────────────────────────────
# Take T = longest market expiry; the surrogate generates paths to T, and we
# read intermediate V[t_idx] for shorter maturities.
H        = 0.10
RHO      = -0.85
N        = 252
T_MAX    = max(T_years_list)
DT       = T_MAX / N

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Surrogate: H={H}, ρ={RHO}, T_max={T_MAX:.4f}y, N={N}, dt={DT:.5f}")

# Index in V (0..N) corresponding to each market maturity
T_idx_arr = torch.tensor(
    [min(int(round(T_i / DT)), N) for T_i in all_T], device=device)
K_arr_t = torch.tensor(K_norm, dtype=torch.float32, device=device)
C_t     = torch.tensor(C_norm, dtype=torch.float32, device=device)

# ATM-IV-implied prior on V_0 (use the shortest-maturity ATM IV)
short_idx = np.argmin(np.abs(all_T - min(T_years_list)))
ATM_short = next(iv for K, T_i, iv in zip(all_K, all_T, all_IVmkt)
                  if abs(T_i - min(T_years_list)) < 1e-6 and abs(K - S0) < 5)
V_PRIOR   = float(ATM_short ** 2)
print(f"  Short-maturity ATM IV = {ATM_short*100:.2f}%  →  V_0 prior ≈ {V_PRIOR:.4f}")

model = PowerVolterraSurrogate(
    N=N, dt=DT, H=H, V_0=V_PRIOR,
    theta_0=V_PRIOR, kappa_0=2.0, nu_0=2.0,
    beta_0=0.5, V_min=1e-6).to(device)
print(f"  Init: θ={model.theta.item():.4f}, c_d={model.c_drift.item():.3e}, "
      f"c_s={model.c_stoch.item():.3e}, β={model.beta.item():.3f}")


# ─── 3.  Training ─────────────────────────────────────────────────────────────
print("\n--- Training to match SPY mid prices across 3 maturities ---")
N_TRAIN, EPOCHS, LR = 128, 500, 1e-3
torch.manual_seed(42)
z_train  = torch.randn(N_TRAIN, N, device=device)
z_perp_t = torch.randn(N_TRAIN, N, device=device)

def surr_prices_multi(model):
    V    = model(z_train)
    V_l  = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z_train + np.sqrt(1 - RHO ** 2) * z_perp_t
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S   = torch.cat([torch.zeros(V.shape[0], 1, device=device),
                          log_inc.cumsum(dim=1)], dim=1)
    S = torch.exp(log_S)
    # Vectorised over (T, K)
    C = torch.zeros(N_OPT, device=device)
    for i in range(N_OPT):
        S_T_i = S[:, T_idx_arr[i]]
        C[i]  = (S_T_i - K_arr_t[i]).clamp(min=0.0).mean()
    return C

opt = torch.optim.Adam(model.parameters(), lr=LR)
best_L, best_state, best_ep = float('inf'), None, 0
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    C_s = surr_prices_multi(model)
    L   = F.mse_loss(C_s, C_t)
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if ep >= 30 and L.item() < best_L:
        best_L = L.item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_ep = ep
    if ep % 50 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep:4d}  L {L.item():.2e}  θ={model.theta.item():.4f}  "
              f"β={model.beta.item():.3f}")
if best_state is not None:
    print(f"  Best Adam L = {best_L:.2e} at ep {best_ep}; restoring")
    model.load_state_dict(best_state)
print(f"  Adam time: {time.time() - t0:.1f} s")

# ─── LBFGS polish step ───────────────────────────────────────────────────────
print("\n--- LBFGS polish (40 iters, strong-Wolfe line search) ---")
lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.3, max_iter=40,
                          history_size=20, line_search_fn='strong_wolfe')

def closure():
    lbfgs.zero_grad()
    L = F.mse_loss(surr_prices_multi(model), C_t)
    L.backward()
    return L

t1 = time.time()
L_lbfgs = lbfgs.step(closure)
print(f"  Post-LBFGS L: {L_lbfgs.item():.2e}  "
      f"(Adam best was {best_L:.2e})")
print(f"  LBFGS time: {time.time() - t1:.1f} s")
print(f"  Final  θ={model.theta.item():.4f}  c_d={model.c_drift.item():.3e}  "
      f"c_s={model.c_stoch.item():.3e}  β={model.beta.item():.3f}")


# ─── 4.  Evaluate ─────────────────────────────────────────────────────────────
print("\n--- Evaluating fit on full multi-maturity surface ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_eval = torch.randn(5_000, N, device=device)
    z_perp_eval = torch.randn(5_000, N, device=device)
    V    = model(z_eval)
    V_l  = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z_eval + np.sqrt(1 - RHO ** 2) * z_perp_eval
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S   = torch.cat([torch.zeros(V.shape[0], 1, device=device),
                          log_inc.cumsum(dim=1)], dim=1)
    S_eval  = torch.exp(log_S).cpu().numpy()

C_surr = np.zeros(N_OPT)
for i in range(N_OPT):
    idx = int(T_idx_arr[i].item())
    C_surr[i] = np.mean(np.maximum(S_eval[:, idx] - K_norm[i], 0.0))
C_surr_usd = C_surr * S0
IV_surr = np.array([implied_vol_single(c, S0, K, 0.0, T_i)
                    for c, K, T_i in zip(C_surr_usd, all_K, all_T)])
IV_market_rec = np.array([implied_vol_single(c, S0, K, 0.0, T_i)
                          for c, K, T_i in zip(all_C, all_K, all_T)])


# ─── 5.  Summary by maturity ──────────────────────────────────────────────────
print("\n" + "=" * 92)
print(f"    MULTI-MATURITY SPY CALIBRATION  (S0=${S0:.2f}, ρ={RHO})")
print("=" * 92)

iv_diffs_all = []
for j, (exp_str, T_i) in enumerate(zip(chosen_exps, T_years_list)):
    mask = np.isclose(all_T, T_i)
    print(f"\n  --- {exp_str}  (T = {T_i:.4f} yr = {days_to(exp_str)} days) ---")
    print(f"  {'K':>8} {'K/S0':>6} {'C_mkt':>8} {'C_surr':>8} "
          f"{'IV_mkt':>7} {'IV_surr':>7} {'IV err (vp)':>11}")
    print("  " + "-" * 70)
    iv_diffs_T = []
    for K, C_m, C_s, iv_m, iv_s in zip(
            all_K[mask], all_C[mask], C_surr_usd[mask],
            IV_market_rec[mask], IV_surr[mask]):
        if np.isfinite(iv_m) and np.isfinite(iv_s):
            d = (iv_s - iv_m) * 100
            iv_diffs_T.append(d)
            iv_diffs_all.append(d)
        else:
            d = np.nan
        print(f"  ${K:6.2f} {K/S0:.3f} ${C_m:6.2f} ${C_s:6.2f} "
              f"{iv_m*100:6.2f}% {iv_s*100:6.2f}% {d:+9.2f}")
    if iv_diffs_T:
        print(f"  → maturity IV RMSE: {np.sqrt(np.mean(np.array(iv_diffs_T)**2)):.2f} vp")

iv_diffs_all = np.array(iv_diffs_all)
print("\n" + "=" * 92)
print(f"  OVERALL IV RMSE: {np.sqrt(np.mean(iv_diffs_all**2)):.2f} vp"
      f"  (mean abs: {np.mean(np.abs(iv_diffs_all)):.2f} vp)"
      f"  (N = {len(iv_diffs_all)} points)")
print("=" * 92)

np.savez(RESULTS / 'spx_calibration_multi.npz',
         S0=S0, all_T=all_T, all_K=all_K, all_C=all_C,
         IV_market=IV_market_rec, IV_surr=IV_surr, C_surr=C_surr_usd,
         theta_final=model.theta.item(), beta_final=model.beta.item(),
         c_drift_final=model.c_drift.item(), c_stoch_final=model.c_stoch.item(),
         chosen_exps=chosen_exps)
print(f"\nSaved → {RESULTS / 'spx_calibration_multi.npz'}")
