"""
power_volterra_demo.py  —  non-affine experiment + joint training.

Two parts:
  A. Non-affine: simulate "rough lognormal" (beta=1) reference, show the
     same Volterra architecture with a learnable V^beta diffusion recovers
     beta ≈ 1 starting from beta_init = 0.5 (Heston).
  B. Joint training: ALL parameters (kernel weights w_i, c_drift, c_stoch,
     theta, beta) trained from a misspecified init, on a rough Heston
     reference, with the same price + V-moment loss as before.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rough_heston_variance
from power_volterra import PowerVolterraSurrogate, simulate_rough_volterra
from implied_vol import (compute_iv_surface, compute_delta_surface,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT, RESULTS)

# ─── Common parameters ────────────────────────────────────────────────────────
H, KAPPA, THETA, NU, V0 = 0.10, 2.0, 0.04, 1.9, 0.04
RHO                     = -0.9
N, T                    = 252, 1.0
DT                      = T / N
N_REF                   = 8_000
N_TRAIN                 = 128
N_EVAL                  = 5_000
EPOCHS                  = 200
LR                      = 1.5e-3
GRAD_CLIP               = 1.0
V_MIN                   = 1.0e-6
LAMBDA_M                = 50.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


def rmse_vp(a, b):
    d = (a - b) * 100
    m = np.isfinite(d)
    return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.nan


def build_call_prices(V, Z, dt, rho, seed):
    """Reference call prices on the standard (T, K) grid."""
    rng    = np.random.default_rng(seed)
    Z_perp = rng.standard_normal(Z.shape)
    Z_v    = Z / (Z.std() + 1e-12)
    dZ_S   = rho * Z_v + np.sqrt(1 - rho ** 2) * Z_perp
    V_l    = V[:, :-1].clip(0)
    log_inc = -0.5 * V_l * dt + np.sqrt(V_l * dt) * dZ_S
    n      = V.shape[0]
    log_S  = np.concatenate([np.zeros((n, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    S      = np.exp(log_S)
    C = np.zeros((len(_MATURITIES_DEFAULT), len(_STRIKES_DEFAULT)))
    for i, Tv in enumerate(_MATURITIES_DEFAULT):
        idx = min(int(round(Tv / dt)), V.shape[1] - 1)
        for j, K in enumerate(_STRIKES_DEFAULT):
            C[i, j] = np.mean(np.maximum(S[:, idx] - K, 0.0))
    return C


T_idx = torch.tensor(
    [min(int(round(Tv / DT)), N) for Tv in _MATURITIES_DEFAULT], device=device)
K_arr = torch.tensor(_STRIKES_DEFAULT, dtype=torch.float32, device=device)


def forward_with_moments(model, z, z_perp):
    V    = model(z)
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


def train_model(model, V_ref, Z_ref, C_ref_t, epochs=EPOCHS, lr=LR, label=""):
    V_mean_ref = torch.tensor(V_ref.mean(0), dtype=torch.float32, device=device)
    V_var_ref  = torch.tensor(V_ref.var(0),  dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(42)
    z       = torch.randn(N_TRAIN, N, device=device)
    z_perp  = torch.randn(N_TRAIN, N, device=device)
    losses_p, losses_m = [], []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        C_s, m_s, v_s = forward_with_moments(model, z, z_perp)
        L_p = F.mse_loss(C_s, C_ref_t)
        L_m = F.mse_loss(m_s, V_mean_ref) + F.mse_loss(v_s, V_var_ref)
        L = L_p + LAMBDA_M * L_m
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        losses_p.append(L_p.item())
        losses_m.append(L_m.item())
        if ep % 50 == 0 or ep == epochs - 1:
            extra = f"β={model.beta.item():.3f} " if hasattr(model, 'beta') else ''
            print(f"  [{label}] ep {ep:4d} L_p {L_p.item():.2e}  L_m {L_m.item():.2e}  "
                  f"θ={model.theta.item():.4f} {extra}")
    print(f"  Training time: {time.time() - t0:.1f} s")
    return losses_p, losses_m


def eval_model(model, V_ref, iv_ref, delta_ref):
    torch.manual_seed(7)
    model.eval()
    with torch.no_grad():
        z = torch.randn(N_EVAL, N, device=device)
        V_surr = model(z).cpu().numpy()
        z_np   = z.cpu().numpy()
    iv, _, _ = compute_iv_surface(V_surr, DT, innovations=z_np, rho=RHO, seed=20)
    delta    = compute_delta_surface(V_surr, z_np, DT, rho=RHO, seed=21)
    j_atm = np.argmin(np.abs(_STRIKES_DEFAULT - 1.0))
    return dict(iv=iv, delta=delta, rmse=rmse_vp(iv_ref, iv),
                bias_T05=delta[1, j_atm] - delta_ref[1, j_atm],
                bias_T025=delta[0, j_atm] - delta_ref[0, j_atm],
                bias_T1=delta[2, j_atm] - delta_ref[2, j_atm])


# ============================================================================
# A.  NON-AFFINE EXPERIMENT
# ============================================================================
print("=" * 78)
print("    PART A  —  non-affine model: rough lognormal (β = 1.0)")
print("=" * 78)

# Reference: rough lognormal (β = 1.0)
V_ln, Z_ln, _ = simulate_rough_volterra(H, KAPPA, THETA, NU, V0, N, N_REF, T,
                                         beta=1.0, seed=1)
print(f"  Reference (β=1):  E[V]={V_ln.mean():.5f}  Var[V]={V_ln.var():.5f}")

iv_ln    = compute_iv_surface(V_ln, DT, innovations=Z_ln, rho=RHO, seed=2)[0]
delta_ln = compute_delta_surface(V_ln, Z_ln, DT, rho=RHO, seed=3)
C_ln_t   = torch.tensor(build_call_prices(V_ln, Z_ln, DT, RHO, seed=2),
                         dtype=torch.float32, device=device)

# Surrogate at WRONG β = 0.5 (Heston), no training — baseline mismatch
print("\n--- PowerVolterra at β=0.5 (Heston), NO training (mismatched arch) ---")
m_h = PowerVolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                              theta_0=THETA, kappa_0=KAPPA, nu_0=NU,
                              beta_0=0.5, V_min=V_MIN).to(device)
r_h = eval_model(m_h, V_ln, iv_ln, delta_ln)
print(f"  IV RMSE {r_h['rmse']:.2f} vp   ATM Δ bias T=.5 {r_h['bias_T05']:+.4f}")

# Train with β learnable, init at 0.5; price + moment loss
print("\n--- PowerVolterra with learnable β (init=0.5), trained on rough lognormal ---")
m_t = PowerVolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                              theta_0=THETA, kappa_0=KAPPA, nu_0=NU,
                              beta_0=0.5, V_min=V_MIN).to(device)
train_model(m_t, V_ln, Z_ln, C_ln_t, label="A")
r_t = eval_model(m_t, V_ln, iv_ln, delta_ln)
print(f"\n  Learned β = {m_t.beta.item():.4f}  (target 1.0)")
print(f"  IV RMSE {r_t['rmse']:.2f} vp   ATM Δ bias T=.5 {r_t['bias_T05']:+.4f}")


# ============================================================================
# B.  JOINT TRAINING (all params learnable, wrong init, rough Heston ground truth)
# ============================================================================
print("\n" + "=" * 78)
print("    PART B  —  joint training of all params, rough Heston reference")
print("=" * 78)

V_rh, Z_rh, _ = simulate_rough_heston_variance(
    H, KAPPA, THETA, NU, V0, N, N_REF, T, seed=1)
iv_rh    = compute_iv_surface(V_rh, DT, innovations=Z_rh, rho=RHO, seed=2)[0]
delta_rh = compute_delta_surface(V_rh, Z_rh, DT, rho=RHO, seed=3)
C_rh_t   = torch.tensor(build_call_prices(V_rh, Z_rh, DT, RHO, seed=2),
                         dtype=torch.float32, device=device)

# Wrong init across the board: θ=0.10, κ=1.0, ν=0.5, β=0.8 (off from 0.5)
print("\n--- PowerVolterra: WRONG init (θ=0.10, κ=1, ν=0.5, β=0.8) → train ---")
m_j = PowerVolterraSurrogate(N=N, dt=DT, H=H, V_0=V0,
                              theta_0=0.10, kappa_0=1.0, nu_0=0.5,
                              beta_0=0.8, V_min=V_MIN).to(device)
# Also perturb the kernel weights slightly so they need to learn back
with torch.no_grad():
    m_j.w.data += 0.2 * torch.randn_like(m_j.w.data)

train_model(m_j, V_rh, Z_rh, C_rh_t, label="B")
r_j = eval_model(m_j, V_rh, iv_rh, delta_rh)
print(f"\n  Final  θ={m_j.theta.item():.4f}  c_d={m_j.c_drift.item():.3e}  "
      f"c_s={m_j.c_stoch.item():.3e}  β={m_j.beta.item():.3f}")
print(f"  IV RMSE {r_j['rmse']:.2f} vp")
print(f"  ATM Δ bias  T=.25: {r_j['bias_T025']:+.4f}  T=.50: {r_j['bias_T05']:+.4f}  "
      f"T=1.0: {r_j['bias_T1']:+.4f}")


# ============================================================================
# C.  Summary
# ============================================================================
print("\n" + "=" * 90)
print("    POWER-LAW VOLTERRA: NON-AFFINE + JOINT TRAINING")
print("=" * 90)
print(f"{'Setting':<54} | {'IV RMSE':>9} | {'ATM Δ bias (T=.5)':>17}")
print("-" * 90)
print(f"{'A: β=0.5 (Heston), rough lognormal ref, no train':<54} | {r_h['rmse']:>7.2f}vp | {r_h['bias_T05']:>+17.4f}")
print(f"{'A: β learnable, trained on rough lognormal':<54} | {r_t['rmse']:>7.2f}vp | {r_t['bias_T05']:>+17.4f}")
print(f"{'   → learned β = ' + f'{m_t.beta.item():.3f}' + '  (target 1.0)':<54} |   |")
print(f"{'B: WRONG init, joint train (kernel + scalars + β)':<54} | {r_j['rmse']:>7.2f}vp | {r_j['bias_T05']:>+17.4f}")
print(f"{'   → learned β = ' + f'{m_j.beta.item():.3f}' + '  (target 0.5)':<54} |   |")
print("=" * 90)

np.savez(RESULTS / 'power_volterra_demo.npz',
         beta_A=m_t.beta.item(), rmse_A=r_t['rmse'], bias_A=r_t['bias_T05'],
         beta_B=m_j.beta.item(), rmse_B=r_j['rmse'], bias_B=r_j['bias_T05'],
         theta_B=m_j.theta.item(),
         c_drift_B=m_j.c_drift.item(), c_stoch_B=m_j.c_stoch.item())
print(f"\nSaved → {RESULTS / 'power_volterra_demo.npz'}")
