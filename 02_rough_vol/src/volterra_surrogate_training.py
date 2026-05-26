"""
volterra_surrogate_training.py  —  end-to-end training of the Volterra
surrogate from observed call prices.

Closes the §5.4 future-work item (i): can the VolterraSurrogate
parameters (theta, c_drift, c_stoch, kernel weights) be LEARNED from
data, rather than initialised at known rough-Heston values?

Setup
-----
  1. Compute reference call prices C_ref(K, T) once from rough Heston.
  2. Initialise VolterraSurrogate at PERTURBED parameters (wrong theta,
     wrong nu) — the surrogate produces a misspecified IV surface.
  3. Train end-to-end on common-random-number MC: gradient flows back
     from call-price MSE through the autoregressive Volterra update.
  4. Evaluate: did training recover the rough Heston IV / delta?
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
N_TRAIN_PATHS           = 128        # tight batch for 2GB GPU + autograd
N_EVAL_PATHS            = 5_000
EPOCHS                  = 150
LR                      = 1.5e-3
GRAD_CLIP               = 1.0
V_MIN                   = 1.0e-6      # avoid sqrt(0) blow-up

# Wrong initialisation — let's see if training recovers
THETA_WRONG = 0.100       # 2.5x target — large vol bias
NU_WRONG    = 0.50        # 4x lower — small vol-of-vol
KAPPA_WRONG = 1.0         # half target

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ─── 1.  Rough Heston reference: paths and call prices ────────────────────────
print("--- Rough Heston reference paths ---")
V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)
iv_ref    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_ref = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)

# Reference call prices: build stock paths and compute prices at (K, T)
rng_ref   = np.random.default_rng(2)
Z_perp_ref = rng_ref.standard_normal((N_REF, N))
Z_v        = Z_rh / (Z_rh.std() + 1e-12)
dZ_S_ref   = RHO * Z_v + np.sqrt(1 - RHO**2) * Z_perp_ref

V_left = V_rh[:, :-1].clip(0)
log_inc = -0.5 * V_left * DT + np.sqrt(V_left * DT) * dZ_S_ref
log_S = np.concatenate([np.zeros((N_REF, 1)), np.cumsum(log_inc, axis=1)], axis=1)
S_ref = np.exp(log_S)

C_ref = np.zeros((len(_MATURITIES_DEFAULT), len(_STRIKES_DEFAULT)))
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    idx = min(int(round(Tv / DT)), V_rh.shape[1] - 1)
    for j, K in enumerate(_STRIKES_DEFAULT):
        C_ref[i, j] = np.mean(np.maximum(S_ref[:, idx] - K, 0.0))
print(f"  C_ref(K=1, T=0.5) = {C_ref[1, 4]:.5f}")
print(f"  C_ref shape: {C_ref.shape}  (maturities x strikes)")
C_ref_t = torch.tensor(C_ref, dtype=torch.float32, device=device)

# Precompute maturity indices
T_idx = torch.tensor(
    [min(int(round(Tv / DT)), N) for Tv in _MATURITIES_DEFAULT],
    device=device)
K_arr = torch.tensor(_STRIKES_DEFAULT, dtype=torch.float32, device=device)


# ─── 2.  Initialise VolterraSurrogate at WRONG parameters ─────────────────────
print(f"\n--- VolterraSurrogate (wrong init: θ={THETA_WRONG}, ν={NU_WRONG}, κ={KAPPA_WRONG}) ---")
model = VolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                          theta_0=THETA_WRONG, kappa_0=KAPPA_WRONG, nu_0=NU_WRONG,
                          V_min=V_MIN).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def surrogate_call_prices(model, z_train, z_perp):
    """Differentiable call-price tensor (n_T, n_K) given fixed noise."""
    V    = model(z_train)                          # (B, N+1)
    V_l  = V[:, :-1].clamp(min=0.0)
    dZ_S = RHO * z_train + np.sqrt(1 - RHO**2) * z_perp
    log_inc = -0.5 * V_l * DT + torch.sqrt(V_l * DT) * dZ_S
    log_S0  = torch.zeros(V.shape[0], 1, device=device)
    log_S   = torch.cat([log_S0, log_inc.cumsum(dim=1)], dim=1)
    S = torch.exp(log_S)                            # (B, N+1)
    C = torch.zeros(len(_MATURITIES_DEFAULT), len(_STRIKES_DEFAULT), device=device)
    for i in range(len(_MATURITIES_DEFAULT)):
        S_T = S[:, T_idx[i]]
        for j in range(len(_STRIKES_DEFAULT)):
            C[i, j] = (S_T - K_arr[j]).clamp(min=0.0).mean()
    return C


# ─── 3.  Training loop (common random numbers) ────────────────────────────────
print("\n--- Training (CRN, gradient flow through Volterra update) ---")
torch.manual_seed(42)
z_train   = torch.randn(N_TRAIN_PATHS, N, device=device)
z_perp_t  = torch.randn(N_TRAIN_PATHS, N, device=device)

losses = []
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    C_surr = surrogate_call_prices(model, z_train, z_perp_t)
    loss   = F.mse_loss(C_surr, C_ref_t)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    losses.append(loss.item())
    if ep % 20 == 0 or ep == EPOCHS - 1:
        # Report current parameter values
        print(f"  ep {ep:4d}  loss {loss.item():.2e}  "
              f"θ={model.theta.item():.4f}  c_s={model.c_stoch.item():.3e}  "
              f"c_d={model.c_drift.item():.3e}")
print(f"  Training time: {time.time() - t0:.1f} s")


# ─── 4.  Evaluation: IV / delta after training ────────────────────────────────
print("\n--- Evaluation (5k MC paths, ρ=−0.9) ---")
torch.manual_seed(7)
model.eval()
with torch.no_grad():
    z_eval = torch.randn(N_EVAL_PATHS, N, device=device)
    V_eval = model(z_eval).cpu().numpy()
    z_np   = z_eval.cpu().numpy()

iv_V,  _, _ = compute_iv_surface(V_eval, DT, innovations=z_np, rho=RHO, seed=20)
delta_V     = compute_delta_surface(V_eval, z_np, DT, rho=RHO, seed=21)


def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan

j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
print("\n" + "=" * 86)
print("    END-TO-END TRAINED VOLTERRA SURROGATE  (rough Heston, ρ=−0.9)")
print("=" * 86)
print(f"  Init params:   θ={THETA_WRONG}, ν={NU_WRONG}, κ={KAPPA_WRONG}  (target θ={THETA}, ν={NU}, κ={KAPPA})")
print(f"  Final params:  θ={model.theta.item():.4f}, "
      f"c_s={model.c_stoch.item():.3e}, c_d={model.c_drift.item():.3e}")
print()
print(f"{'Condition':<42} | {'IV RMSE (vp)':>12} | {'ATM Δ (T=.50)':>13} | {'bias':>8}")
print("-" * 86)
ref50 = delta_ref[1, j_atm]
print(f"{'Rough Heston reference (ρ=−0.9)':<42} | {' —':>12} | {ref50:13.4f} | {' —':>8}")
print(f"{'Best spectral (causal SF rBergomi-V)':<42} | {4.57:>12.2f} | {0.6247:13.4f} | {-0.150:+8.4f}")
print(f"{'Volterra surrogate (RH-initialised)':<42} | {0.80:>12.2f} | {0.7765:13.4f} | {+0.002:+8.4f}")
rmse_V = rmse_vp(iv_ref, iv_V)
dval   = delta_V[1, j_atm]
print(f"{'Volterra surrogate (WRONG init → trained)':<42} | {rmse_V:>12.2f} | {dval:13.4f} | {dval-ref50:+8.4f}")
print("=" * 86)

print("\n--- Loss curve (sample) ---")
for ep in [0, 5, 10, 20, 50, 100, 199]:
    if ep < len(losses):
        print(f"  ep {ep:3d}  loss {losses[ep]:.2e}")

print("\n--- Per-maturity ATM deltas after training ---")
print(f"{'T':>6}  {'Δ_ref':>8}  {'Δ_V':>8}  {'bias':>8}")
for i, Tv in enumerate(_MATURITIES_DEFAULT):
    r  = delta_ref[i, j_atm]
    dV = delta_V[i, j_atm]
    print(f"  {Tv:.2f}  {r:8.4f}  {dV:8.4f}  {dV-r:+8.4f}")

np.savez(RESULTS / 'volterra_surrogate_training.npz',
         iv_ref=iv_ref, iv_V=iv_V,
         delta_ref=delta_ref, delta_V=delta_V,
         losses=np.array(losses),
         theta_final=model.theta.item(),
         c_drift_final=model.c_drift.item(),
         c_stoch_final=model.c_stoch.item())
print(f"\nSaved → {RESULTS / 'volterra_surrogate_training.npz'}")
