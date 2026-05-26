"""
volterra_training_moments.py  —  auxiliary moment-matching loss for Volterra training.

Question:  the §5.4 price-only training hit an identifiability wall at
ATM Δ bias −0.09 (5.81 vp) — many parameter sets give nearly identical
call prices.  Does adding an auxiliary V-moment loss
   L_total = L_price + λ_m * L_moments
break the degeneracy and recover rough Heston more faithfully?

Auxiliary loss:
   L_moments = sum_t [ (E[V_surr_t] - E[V_ref_t])^2
                       + (Var[V_surr_t] - Var[V_ref_t])^2 ]

Both moments are evaluated on the same MC batch as the price loss
(common random numbers).
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance
from volterra_surrogate import VolterraSurrogate
from implied_vol import (compute_iv_surface, compute_delta_surface,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT, RESULTS)

# ─── Parameters ───────────────────────────────────────────────────────────────
H, KAPPA, THETA, NU, V0 = 0.10, 2.0, 0.04, 1.9, 0.04
RHO                     = -0.9
N, T                    = 252, 1.0
DT                      = T / N
N_REF                   = 8_000
N_TRAIN_PATHS           = 128
N_EVAL_PATHS            = 5_000
EPOCHS                  = 200
LR                      = 1.5e-3
GRAD_CLIP               = 1.0
V_MIN                   = 1.0e-6
LAMBDA_MOMENT           = 50.0       # weight on auxiliary moment loss

# Same heavy misspecification as the price-only baseline
THETA_WRONG = 0.100
NU_WRONG    = 0.50
KAPPA_WRONG = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  Reference: paths, call prices, V moments ────────────────────────────
print("--- Rough Heston reference ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)
iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)

rng_ref    = np.random.default_rng(2)
Z_perp_ref = rng_ref.standard_normal((N_REF, N))
Z_v        = Z_rh / (Z_rh.std() + 1e-12)
dZ_S_ref   = RHO * Z_v + np.sqrt(1 - RHO**2) * Z_perp_ref

V_left  = V_rh[:, :-1].clip(0)
log_inc = -0.5 * V_left * DT + np.sqrt(V_left * DT) * dZ_S_ref
log_S   = np.concatenate([np.zeros((N_REF, 1)), np.cumsum(log_inc, axis=1)], axis=1)
S_ref   = np.exp(log_S)

C_ref = np.zeros((len(_MATURITIES_DEFAULT), len(_STRIKES_DEFAULT)))
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    idx = min(int(round(Tv / DT)), V_rh.shape[1] - 1)
    for j, K in enumerate(_STRIKES_DEFAULT):
        C_ref[i, j] = np.mean(np.maximum(S_ref[:, idx] - K, 0.0))
C_ref_t = torch.tensor(C_ref, dtype=torch.float32, device=device)

# V moments at every step (reference targets)
V_mean_ref = torch.tensor(V_rh.mean(axis=0), dtype=torch.float32, device=device)  # (N+1,)
V_var_ref  = torch.tensor(V_rh.var(axis=0),  dtype=torch.float32, device=device)  # (N+1,)

T_idx = torch.tensor(
    [min(int(round(Tv / DT)), N) for Tv in _MATURITIES_DEFAULT], device=device)
K_arr = torch.tensor(_STRIKES_DEFAULT, dtype=torch.float32, device=device)


# ─── 2.  Model + optimiser ───────────────────────────────────────────────────
print(f"\n--- Volterra (wrong init θ={THETA_WRONG}, ν={NU_WRONG}, κ={KAPPA_WRONG}) ---")
model = VolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                          theta_0=THETA_WRONG, kappa_0=KAPPA_WRONG, nu_0=NU_WRONG,
                          V_min=V_MIN).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def forward_with_moments(model, z, z_perp):
    """Return (call_prices, V_mean_path, V_var_path) — all differentiable."""
    V    = model(z)                                  # (B, N+1)
    V_l  = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z + np.sqrt(1 - RHO ** 2) * z_perp
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S0  = torch.zeros(V.shape[0], 1, device=device)
    log_S   = torch.cat([log_S0, log_inc.cumsum(dim=1)], dim=1)
    S = torch.exp(log_S)
    C = torch.zeros(len(_MATURITIES_DEFAULT), len(_STRIKES_DEFAULT), device=device)
    for i in range(len(_MATURITIES_DEFAULT)):
        S_T = S[:, T_idx[i]]
        for j in range(len(_STRIKES_DEFAULT)):
            C[i, j] = (S_T - K_arr[j]).clamp(min=0.0).mean()
    return C, V.mean(dim=0), V.var(dim=0)


# ─── 3.  Training (price + moment loss, CRN) ─────────────────────────────────
print(f"\n--- Training with auxiliary moment loss (λ={LAMBDA_MOMENT}) ---")
torch.manual_seed(42)
z_train  = torch.randn(N_TRAIN_PATHS, N, device=device)
z_perp_t = torch.randn(N_TRAIN_PATHS, N, device=device)

losses_price = []
losses_mmt   = []
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    C_surr, V_mean_s, V_var_s = forward_with_moments(model, z_train, z_perp_t)
    L_price = F.mse_loss(C_surr, C_ref_t)
    L_mmt   = F.mse_loss(V_mean_s, V_mean_ref) + F.mse_loss(V_var_s, V_var_ref)
    L       = L_price + LAMBDA_MOMENT * L_mmt
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    losses_price.append(L_price.item())
    losses_mmt.append(L_mmt.item())
    if ep % 25 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep:4d}  L_price {L_price.item():.2e}  "
              f"L_mmt {L_mmt.item():.2e}  "
              f"θ={model.theta.item():.4f}  c_s={model.c_stoch.item():.3e}")
print(f"  Training time: {time.time() - t0:.1f} s")


# ─── 4.  Evaluation ──────────────────────────────────────────────────────────
print("\n--- Evaluation (5k MC paths) ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_eval = torch.randn(N_EVAL_PATHS, N, device=device)
    V_eval = model(z_eval).cpu().numpy()
    z_np   = z_eval.cpu().numpy()
print(f"  E[V_surr] = {V_eval.mean():.5f}    target (E[V_rh]) = {V_rh.mean():.5f}")
print(f"  Var[V_surr] = {V_eval.var():.5f}  target (Var[V_rh]) = {V_rh.var():.5f}")

iv_V, _, _ = compute_iv_surface(V_eval, DT, innovations=z_np, rho=RHO, seed=20)
delta_V    = compute_delta_surface(V_eval, z_np, DT, rho=RHO, seed=21)


def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan

j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
print("\n" + "=" * 90)
print("    VOLTERRA SURROGATE: PRICE + MOMENT-MATCHING TRAINING")
print("=" * 90)
print(f"  Init: θ={THETA_WRONG}, ν={NU_WRONG}, κ={KAPPA_WRONG}  →  target θ={THETA}, ν={NU}, κ={KAPPA}")
print(f"  λ_moment = {LAMBDA_MOMENT}")
print()
print(f"{'Condition':<48} | {'IV RMSE (vp)':>12} | {'ATM Δ (T=.50)':>13} | {'bias':>8}")
print("-" * 90)
ref50 = delta_ref[1, j_atm]
print(f"{'Rough Heston reference':<48} | {' —':>12} | {ref50:13.4f} | {' —':>8}")
print(f"{'Spectral best (causal SF rBergomi-V)':<48} | {4.57:>12.2f} | {0.6247:13.4f} | {-0.150:+8.4f}")
print(f"{'Volterra RH-init (no training)':<48} | {0.80:>12.2f} | {0.7765:13.4f} | {+0.002:+8.4f}")
print(f"{'Volterra price-only (wrong init→trained)':<48} | {5.81:>12.2f} | {0.6869:13.4f} | {-0.088:+8.4f}")
rmse_V = rmse_vp(iv_ref, iv_V)
dval   = delta_V[1, j_atm]
print(f"{'NEW Volterra price + moment (wrong init)':<48} | {rmse_V:>12.2f} | {dval:13.4f} | {dval-ref50:+8.4f}")
print("=" * 90)

print("\n--- Per-maturity ATM deltas ---")
print(f"{'T':>6}  {'Δ_ref':>8}  {'Δ_V':>8}  {'bias':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    r  = delta_ref[i, j_atm]
    dV = delta_V[i, j_atm]
    print(f"  {Tv:.2f}  {r:8.4f}  {dV:8.4f}  {dV-r:+8.4f}")

print(f"\n  Final learned params: θ={model.theta.item():.4f}  "
      f"c_d={model.c_drift.item():.3e}  c_s={model.c_stoch.item():.3e}")

np.savez(RESULTS / 'volterra_training_moments.npz',
         iv_ref=iv_ref, iv_V=iv_V,
         delta_ref=delta_ref, delta_V=delta_V,
         losses_price=np.array(losses_price),
         losses_mmt=np.array(losses_mmt),
         theta_final=model.theta.item(),
         c_drift_final=model.c_drift.item(),
         c_stoch_final=model.c_stoch.item())
print(f"\nSaved → {RESULTS / 'volterra_training_moments.npz'}")
