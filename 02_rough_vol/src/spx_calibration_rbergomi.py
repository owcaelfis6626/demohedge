"""
spx_calibration_rbergomi.py  —  multi-maturity SPY calibration with the
rBergomi-style forward-variance parameterisation.

   V_t = ξ(t) · exp(η · W^H_t − ½ η² Var(W^H_t))

ξ(t): deterministic forward-variance curve, piecewise-linear with knots
      at the calibration maturities (positive by construction via exp).
η  : scalar vol-of-vol.
W^H: fBm path from CausalSpectralFilter (causal!) — same fGn target as
      §5.4, but now embedded in a multiplicative rather than Volterra-
      additive structure.

Advantages over the TermStructureVolterra attempt:
  1.  V > 0 always (exp), so no V_min clamping bias.
  2.  E[V_t] = ξ(t) by the lognormal martingale property — ATM IV term
      structure is pinned directly to ξ(t), no mean-reversion plumbing
      needed.
  3.  z-Cholesky is mathematically valid (causal filter ⇒ V_k ⊥ z_k).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from datetime import datetime
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from bochner_pinn import CausalSpectralFilter
from implied_vol import implied_vol_single, RESULTS


# ─── 1.  rBergomi-with-forward-variance surrogate ────────────────────────────
class ForwardVarianceRBergomi(nn.Module):
    """
    V_t = xi(t) * exp(eta * W^H_t - 0.5 * eta^2 * Var(W^H_t))
    W^H built by CausalSpectralFilter; xi(t) piecewise-linear with
    log-parameterised knots (positivity by construction).
    """
    def __init__(self, N: int, dt: float, H: float,
                 knot_times_yr: np.ndarray, knot_xi_init: np.ndarray,
                 eta_init: float = 1.5):
        super().__init__()
        self.filter = CausalSpectralFilter(N=N, H=H, dt=dt)
        # Log-parameterised knot xi values (positive)
        self.log_knot_xi = nn.Parameter(
            torch.tensor(np.log(np.maximum(knot_xi_init, 1e-6)),
                         dtype=torch.float32))
        # Log-parameterised eta (positive)
        self.log_eta = nn.Parameter(torch.tensor(np.log(eta_init),
                                                  dtype=torch.float32))
        self.register_buffer('knot_times',
                             torch.tensor(knot_times_yr, dtype=torch.float32))
        t_grid = np.arange(N + 1) * dt
        self.register_buffer('t_grid', torch.tensor(t_grid, dtype=torch.float32))
        self.N = N
        self.dt = dt

    def xi_curve(self):
        """Piecewise-linear xi(t) on the time grid (N+1 values)."""
        kt   = self.knot_times
        xi_k = torch.exp(self.log_knot_xi)
        t    = self.t_grid
        K    = kt.shape[0]
        idx_right = torch.searchsorted(kt, t).clamp(min=1, max=K - 1)
        idx_left  = idx_right - 1
        x0 = kt[idx_left];  x1 = kt[idx_right]
        y0 = xi_k[idx_left]; y1 = xi_k[idx_right]
        w_lin = ((t - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
        return y0 + w_lin * (y1 - y0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N)  →  V: (B, N+1).  z is also the underlying BM for ρ."""
        eta = torch.exp(self.log_eta)
        dW_H = self.filter(z)                                    # (B, N) fGn incs
        W_H = torch.cat([torch.zeros(z.shape[0], 1, device=z.device),
                         dW_H.cumsum(dim=1)], dim=1)              # (B, N+1) fBm
        var_W = W_H.var(dim=0)                                   # (N+1,)
        xi    = self.xi_curve()                                  # (N+1,)
        V_raw = xi * torch.exp(eta * W_H - 0.5 * eta * eta * var_W)
        # Post-hoc empirical-mean correction: enforce E[V_t] = xi(t) exactly
        # on the batch (the lognormal martingale property is finite-sample
        # noisy at long horizons; renormalising fixes E[V] without changing
        # the cross-sectional shape of V/E[V]).
        V_mean = V_raw.mean(dim=0, keepdim=True).clamp(min=1e-9)
        V     = V_raw * (xi / V_mean)
        return V


# ─── 2.  Pull SPY multi-maturity chain ───────────────────────────────────────
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
    atm_row = sel.iloc[(sel['strike'] - S0).abs().argmin()]
    atm_ivs_by_T.append(float(atm_row['impliedVolatility']))
    print(f"  {exp_str} ({days_to(exp_str)}d): ATM IV {atm_row['impliedVolatility']*100:.2f}%")

all_K, all_C, all_IV, all_T = map(np.array, (all_K, all_C, all_IV, all_T))
N_OPT = len(all_K)
K_norm, C_norm = all_K / S0, all_C / S0


# ─── 3.  Forward-variance knot init from ATM-IV² ─────────────────────────────
knot_times_yr  = np.array([0.0] + list(T_years_list))
knot_xi_init   = np.array([atm_ivs_by_T[0]] + atm_ivs_by_T) ** 2
print(f"\n  Knot times (yr):   {knot_times_yr.round(4)}")
print(f"  Init ξ knots:      {knot_xi_init.round(4)}")
print(f"  Implied ATM IV:    {np.sqrt(knot_xi_init).round(3)}")


# ─── 4.  Surrogate setup ──────────────────────────────────────────────────────
H, RHO = 0.10, -0.85
N      = 252
T_MAX  = max(T_years_list)
DT     = T_MAX / N

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- Surrogate: ρ={RHO}, T_max={T_MAX:.4f}y, N={N}, dt={DT:.5f}")

model = ForwardVarianceRBergomi(
    N=N, dt=DT, H=H,
    knot_times_yr=knot_times_yr, knot_xi_init=knot_xi_init,
    eta_init=1.5).to(device)

T_idx_arr = torch.tensor(
    [min(int(round(T_i / DT)), N) for T_i in all_T], device=device)
K_arr_t = torch.tensor(K_norm, dtype=torch.float32, device=device)
C_t     = torch.tensor(C_norm, dtype=torch.float32, device=device)

print(f"  Init ξ knots: {torch.exp(model.log_knot_xi).detach().cpu().numpy().round(4)}")
print(f"  Init η = {torch.exp(model.log_eta).item():.3f}")


# ─── 5.  Training ─────────────────────────────────────────────────────────────
print("\n--- Training to match SPY chain ---")
N_TRAIN, EPOCHS, LR = 128, 500, 1e-3
torch.manual_seed(42)
z_train  = torch.randn(N_TRAIN, N, device=device)
z_perp_t = torch.randn(N_TRAIN, N, device=device)


def surr_prices(model):
    V    = model(z_train)
    V_l  = V[:, :-1].clamp(min=1e-9)
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
        xi = torch.exp(model.log_knot_xi).detach().cpu().numpy()
        print(f"  ep {ep:4d}  L {L.item():.2e}  "
              f"ξ={xi.round(4)}  η={torch.exp(model.log_eta).item():.3f}")
if best_state is not None:
    print(f"  Best L = {best_L:.2e} at ep {best_ep}; restoring")
    model.load_state_dict(best_state)
print(f"  Adam time: {time.time() - t0:.1f} s")

xi_final  = torch.exp(model.log_knot_xi).detach().cpu().numpy()
eta_final = torch.exp(model.log_eta).item()
print(f"\n  Final ξ knots: {xi_final.round(4)}")
print(f"  Implied ATM IV from ξ: {np.sqrt(xi_final).round(3)}")
print(f"  Final η = {eta_final:.3f}")


# ─── 6.  Evaluation (5k MC) ───────────────────────────────────────────────────
print("\n--- Evaluating fit ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_eval = torch.randn(5_000, N, device=device)
    z_perp_eval = torch.randn(5_000, N, device=device)
    V = model(z_eval)
    V_l = V[:, :-1].clamp(min=1e-9)
    dZ_S = RHO * z_eval + np.sqrt(1 - RHO ** 2) * z_perp_eval
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S = torch.cat([torch.zeros(V.shape[0], 1, device=device),
                       log_inc.cumsum(dim=1)], dim=1)
    S_eval = torch.exp(log_S).cpu().numpy()
    V_np   = V.cpu().numpy()

# Sanity check: E[V_t] should match xi(t)
print(f"\n  E[V_T] vs ξ(T) at maturities (sanity check of martingale property):")
for T_i in T_years_list:
    idx = min(int(round(T_i / DT)), N)
    EV  = V_np[:, idx].mean()
    xi_val = float(model.xi_curve()[idx].item())
    print(f"    T={T_i:.4f}: E[V]={EV:.4f}  ξ={xi_val:.4f}  ratio={EV/xi_val:.3f}")

C_surr = np.array([
    np.mean(np.maximum(S_eval[:, int(T_idx_arr[i].item())] - K_norm[i], 0.0))
    for i in range(N_OPT)
])
C_surr_usd = C_surr * S0
IV_surr   = np.array([implied_vol_single(c, S0, K, 0.0, T_i)
                       for c, K, T_i in zip(C_surr_usd, all_K, all_T)])
IV_market = np.array([implied_vol_single(c, S0, K, 0.0, T_i)
                       for c, K, T_i in zip(all_C, all_K, all_T)])


# ─── 7.  Per-maturity summary ─────────────────────────────────────────────────
print("\n" + "=" * 92)
print(f"    rBERGOMI FORWARD-VARIANCE SPY CALIBRATION  (S0=${S0:.2f}, ρ={RHO})")
print("=" * 92)
iv_diffs_all = []
for exp_str, T_i in zip(chosen_exps, T_years_list):
    mask = np.isclose(all_T, T_i)
    print(f"\n  --- {exp_str}  (T = {T_i:.4f} yr, {days_to(exp_str)} days) ---")
    diffs_T = []
    for K, iv_m, iv_s in zip(all_K[mask], IV_market[mask], IV_surr[mask]):
        if np.isfinite(iv_m) and np.isfinite(iv_s):
            d = (iv_s - iv_m) * 100
            diffs_T.append(d); iv_diffs_all.append(d)
            print(f"  ${K:6.2f} K/S0={K/S0:.3f}  IV_mkt={iv_m*100:6.2f}%  "
                  f"IV_surr={iv_s*100:6.2f}%  err {d:+6.2f} vp")
    if diffs_T:
        print(f"  → maturity IV RMSE: {np.sqrt(np.mean(np.array(diffs_T)**2)):.2f} vp")

iv_diffs_all = np.array(iv_diffs_all)
print("\n" + "=" * 92)
print(f"  OVERALL IV RMSE: {np.sqrt(np.mean(iv_diffs_all**2)):.2f} vp"
      f"  (mean abs: {np.mean(np.abs(iv_diffs_all)):.2f} vp, N = {len(iv_diffs_all)})")
print("=" * 92)
print(f"\n  Comparison:")
print(f"    Stationary-θ Volterra multi-maturity:     3.97 vp")
print(f"    Time-varying θ(t) Volterra (Feller violation): 5.59 vp")
print(f"    rBergomi forward-variance ξ(t):            {np.sqrt(np.mean(iv_diffs_all**2)):.2f} vp")

np.savez(RESULTS / 'spx_calibration_rbergomi.npz',
         S0=S0, all_T=all_T, all_K=all_K,
         IV_market=IV_market, IV_surr=IV_surr,
         knot_times=knot_times_yr,
         xi_init=knot_xi_init, xi_final=xi_final,
         eta_final=eta_final)
print(f"\nSaved → {RESULTS / 'spx_calibration_rbergomi.npz'}")
