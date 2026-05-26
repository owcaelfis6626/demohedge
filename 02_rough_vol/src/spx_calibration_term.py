"""
spx_calibration_term.py  —  multi-maturity SPY calibration with a
time-varying forward variance curve θ(t).

The §5.4 stationary-θ surrogate hits a 5-vp long-end residual because a
single θ cannot match SPX's rising ATM-IV term structure (15.4% → 18.6%
→ 20.8% at 32, 88, 179 days).  This script replaces the scalar θ with a
piecewise-linear function θ(t) parameterised by knots at the three
calibration maturities (plus t = 0).  The Volterra drift becomes
   c_drift * sum_j w[k-j] * (θ(t_j) - V_j),
giving the model the freedom to track the term structure.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from datetime import datetime
from pathlib import Path
from scipy.special import gamma as _gamma

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from implied_vol import implied_vol_single, RESULTS


# ─── Time-varying-θ Volterra surrogate ───────────────────────────────────────
class TermStructureVolterra(nn.Module):
    """
    Volterra surrogate with piecewise-linear θ(t).

      V_{k+1} = V_0
              + c_drift * sum_j w[k-j] * (θ(t_j) - V_j)
              + c_stoch * sum_j w[k-j] * V_j^β * z_j

    Knots: K positions (incl. 0); knot_thetas are learnable values at
    those knot times.  Linear interpolation between knots.
    """
    def __init__(self, N: int, dt: float, H: float,
                 knot_times_yr: np.ndarray, knot_thetas_init: np.ndarray,
                 V_0: float = None,
                 kappa_0: float = 2.0, nu_0: float = 2.0,
                 beta_0: float = 0.5, V_min: float = 1.0e-6):
        super().__init__()
        alpha  = H + 0.5
        i_arr  = np.arange(N, dtype=np.float64)
        w_init = (i_arr + 1.0) ** (alpha - 1.0)
        self.w       = nn.Parameter(torch.from_numpy(w_init).float())
        self.c_drift = nn.Parameter(
            torch.tensor(kappa_0 * dt ** alpha / _gamma(alpha), dtype=torch.float32))
        self.c_stoch = nn.Parameter(
            torch.tensor(nu_0 * dt ** (alpha - 0.5) / _gamma(alpha), dtype=torch.float32))
        self.beta    = nn.Parameter(torch.tensor(beta_0, dtype=torch.float32))
        # Time-varying θ:  first knot tied to V_0 (frozen); knots 1..K-1
        # parameterised via log to stay strictly positive.
        v0_val = V_0 if V_0 is not None else float(knot_thetas_init[0])
        self.register_buffer('V_0', torch.tensor(v0_val, dtype=torch.float32))
        self.register_buffer('knot_times',
                             torch.tensor(knot_times_yr, dtype=torch.float32))
        # Force knot 0 == V_0, store rest as log_thetas (positive)
        rest_init = np.maximum(knot_thetas_init[1:], 1e-6)
        self.log_thetas_rest = nn.Parameter(
            torch.tensor(np.log(rest_init), dtype=torch.float32))
        # Time grid (immutable)
        t_grid = np.arange(N + 1) * dt
        self.register_buffer('t_grid', torch.tensor(t_grid, dtype=torch.float32))
        self.N     = N
        self.dt    = dt
        self.V_min = V_min

    def knot_thetas(self):
        """Knot θ values: knot 0 = V_0 (frozen), rest from exp(log_thetas)."""
        rest = torch.exp(self.log_thetas_rest)
        return torch.cat([self.V_0.unsqueeze(0), rest])

    def _theta_curve(self):
        """Piecewise-linear interpolation on the time grid."""
        t = self.t_grid
        K = self.knot_times.shape[0]
        kt = self.knot_thetas()
        idx_right = torch.searchsorted(self.knot_times, t).clamp(min=1, max=K - 1)
        idx_left  = idx_right - 1
        x0 = self.knot_times[idx_left]
        x1 = self.knot_times[idx_right]
        y0 = kt[idx_left]
        y1 = kt[idx_right]
        w_lin = ((t - x0) / (x1 - x0).clamp(min=1e-9)).clamp(min=0.0, max=1.0)
        return y0 + w_lin * (y1 - y0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        theta_arr = self._theta_curve()                     # (N+1,)
        B = z.shape[0]
        V_list = [self.V_0.expand(B)]
        for k in range(self.N):
            V_curr      = torch.stack(V_list, dim=1)         # (B, k+1)
            wk          = self.w[:k + 1].flip(0)             # (k+1,)
            theta_used  = theta_arr[:k + 1]                  # (k+1,)
            drift_sum   = ((theta_used - V_curr) * wk).sum(dim=1)
            V_safe      = V_curr.clamp(min=self.V_min)
            V_beta      = V_safe.pow(self.beta)
            stoch_sum   = (V_beta * z[:, :k + 1] * wk).sum(dim=1)
            V_next      = (self.V_0
                           + self.c_drift * drift_sum
                           + self.c_stoch * stoch_sum).clamp(min=self.V_min)
            V_list.append(V_next)
        return torch.stack(V_list, dim=1)


# ─── 1.  Fetch SPY multi-maturity chain ───────────────────────────────────────
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
T_years_list = [days_to(e) / 365.25 for e in chosen_exps]
print(f"  Spot: ${S0:.2f}; expirations: {chosen_exps}; T={T_years_list}")

all_K, all_C, all_IV, all_T = [], [], [], []
ratio_grid = np.array([0.85, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15])
atm_ivs_by_T = []
for exp_str, T_yr in zip(chosen_exps, T_years_list):
    calls = spy.option_chain(exp_str).calls
    calls = calls[(calls['strike'] >= 0.80 * S0) & (calls['strike'] <= 1.20 * S0)]
    calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
    calls['mid']    = 0.5 * (calls['bid'] + calls['ask'])
    calls['spread'] = (calls['ask'] - calls['bid']) / calls['mid']
    calls = calls[calls['spread'] < 0.20].sort_values('strike').reset_index(drop=True)
    chosen_idx = []
    for K_t in ratio_grid * S0:
        j = (calls['strike'] - K_t).abs().idxmin()
        if j not in chosen_idx:
            chosen_idx.append(j)
    sel = calls.loc[chosen_idx]
    for _, row in sel.iterrows():
        all_K.append(row['strike']); all_C.append(row['mid'])
        all_IV.append(row['impliedVolatility']); all_T.append(T_yr)
    # Record ATM IV at this maturity for knot init
    atm_row = sel.iloc[(sel['strike'] - S0).abs().argmin()]
    atm_ivs_by_T.append(float(atm_row['impliedVolatility']))
    print(f"  {exp_str} ({days_to(exp_str)}d): ATM IV {atm_row['impliedVolatility']*100:.2f}%")

all_K, all_C, all_IV, all_T = map(np.array, (all_K, all_C, all_IV, all_T))
N_OPT = len(all_K)
K_norm, C_norm = all_K / S0, all_C / S0
print(f"  Total points: {N_OPT}")


# ─── 2.  Build θ knots from ATM-IV term structure ────────────────────────────
# Knots at t = 0, T_1, T_2, T_3.  Init θ(t_i) = ATM_IV(t_i)^2.
# θ(0) is set equal to ATM_IV(short)^2.
knot_times_yr     = np.array([0.0] + list(T_years_list))
knot_thetas_init  = np.array([atm_ivs_by_T[0]] + atm_ivs_by_T) ** 2
print(f"\n  Knot times (yr): {knot_times_yr.round(4)}")
print(f"  Init θ knots:    {knot_thetas_init.round(4)}")


# ─── 3.  Surrogate setup ──────────────────────────────────────────────────────
H, RHO = 0.10, -0.85
N      = 252
T_MAX  = max(T_years_list)
DT     = T_MAX / N

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Surrogate: ρ={RHO}, T_max={T_MAX:.4f}y, N={N}, dt={DT:.5f}")

model = TermStructureVolterra(
    N=N, dt=DT, H=H,
    knot_times_yr=knot_times_yr, knot_thetas_init=knot_thetas_init,
    V_0=float(knot_thetas_init[0]),
    kappa_0=2.0, nu_0=2.0, beta_0=0.5, V_min=1e-6).to(device)

T_idx_arr = torch.tensor(
    [min(int(round(T_i / DT)), N) for T_i in all_T], device=device)
K_arr_t = torch.tensor(K_norm, dtype=torch.float32, device=device)
C_t     = torch.tensor(C_norm, dtype=torch.float32, device=device)

print(f"  Init knot θ (tied/positive): {model.knot_thetas().detach().cpu().numpy().round(4)}")
print(f"  Init c_d={model.c_drift.item():.3e}  c_s={model.c_stoch.item():.3e}  "
      f"β={model.beta.item():.3f}")


# ─── 4.  Training (CRN) ───────────────────────────────────────────────────────
print("\n--- Training to match SPY multi-maturity chain ---")
N_TRAIN, EPOCHS, LR = 128, 500, 1e-3
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


opt = torch.optim.Adam(model.parameters(), lr=LR)
best_L, best_state, best_ep = float('inf'), None, 0
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    L = F.mse_loss(surr_prices(model), C_t)
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if ep >= 30 and L.item() < best_L:
        best_L = L.item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_ep = ep
    if ep % 50 == 0 or ep == EPOCHS - 1:
        kts = model.knot_thetas().detach().cpu().numpy().round(4)
        print(f"  ep {ep:4d}  L {L.item():.2e}  knots={kts}  "
              f"β={model.beta.item():.3f}  c_s={model.c_stoch.item():.3e}")
if best_state is not None:
    print(f"  Best L = {best_L:.2e} at ep {best_ep}; restoring")
    model.load_state_dict(best_state)
print(f"  Adam time: {time.time() - t0:.1f} s")

print(f"\n  Final knot θ: {model.knot_thetas().detach().cpu().numpy().round(4)}")
print(f"  Final β = {model.beta.item():.3f}, c_d={model.c_drift.item():.3e}, "
      f"c_s={model.c_stoch.item():.3e}")


# ─── 5.  Evaluate ─────────────────────────────────────────────────────────────
print("\n--- Evaluating fit ---")
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


# ─── 6.  Per-maturity summary ─────────────────────────────────────────────────
print("\n" + "=" * 92)
print(f"    TIME-VARYING θ(t) SPY CALIBRATION  (S0=${S0:.2f}, ρ={RHO})")
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
print(f"\n  Comparison: stationary θ multi-maturity → 3.97 vp")
print(f"              time-varying θ(t)            → {np.sqrt(np.mean(iv_diffs_all**2)):.2f} vp")

np.savez(RESULTS / 'spx_calibration_term.npz',
         S0=S0, all_T=all_T, all_K=all_K,
         IV_market=IV_market, IV_surr=IV_surr,
         knot_times=knot_times_yr,
         knot_thetas_init=knot_thetas_init,
         knot_thetas_final=model.knot_thetas().detach().cpu().numpy(),
         beta_final=model.beta.item(),
         c_drift_final=model.c_drift.item(),
         c_stoch_final=model.c_stoch.item())
print(f"\nSaved → {RESULTS / 'spx_calibration_term.npz'}")
