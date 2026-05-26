"""
surrogate_comparison.py  —  End-to-end IV surface comparison.

MC truth (rBergomi, Wood-Chan fBm) vs three Bochner PINN surrogates:
  1. SpectralFilter  trained on exact Monte Carlo fGn PSD
  2. SpectralFilter  trained on asymptotic C|f|^{1-2H} PSD
  3. BochnerMLP      trained on exact Monte Carlo fGn PSD

Output:
  • RMSE table (vol points) by maturity and overall
  • figures/iv_smile_comparison.png
  • results/surrogate_comparison.npz
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_variance
from bochner_pinn import (SpectralFilter, BochnerMLP, get_exact_fgn_psd,
                           asymptotic_fgn_psd, train_one)
from implied_vol import (compute_iv_surface, compare_iv_surfaces,
                          plot_iv_comparison,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT,
                          RESULTS, FIGURES)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
H, eta, xi0 = 0.1, 1.9, 0.04
N, T  = 252, 1.0
dt    = T / N
N_REF  = 10_000   # MC reference paths (more = lower noise floor)
N_SURR =  5_000   # surrogate paths per model
EPOCHS = 3_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

t_grid = np.linspace(0.0, T, N + 1)
drift  = -0.5 * eta ** 2 * t_grid ** (2 * H)   # (N+1,) Itô correction


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Reference MC IV surface
# ─────────────────────────────────────────────────────────────────────────────
print(f"=== 1. Reference MC IV surface  ({N_REF:,} paths, ρ=0) ===")
V_ref, _, _ = simulate_rbm_variance(H, eta, xi0, N, N_REF, T, seed=1)
print(f"     E[V_ref] = {V_ref.mean():.5f}  (target ξ₀ = {xi0:.4f})")

iv_ref, K_arr, T_arr = compute_iv_surface(V_ref, dt, rho=0.0,
                                            strikes=_STRIKES_DEFAULT,
                                            maturities=_MATURITIES_DEFAULT)
print("     Reference IV (%):")
header = "     T\\K  " + "  ".join(f"{k:.2f}" for k in _STRIKES_DEFAULT)
print(header)
for i, Tv in enumerate(T_arr):
    row = f"     {Tv:.2f}  " + "  ".join(
        f"{v*100:5.2f}" if np.isfinite(v) else "  nan" for v in iv_ref[i])
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Train surrogates
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== 2. Training surrogates  ({EPOCHS} epochs each) ===")
freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
psd_asymp = asymptotic_fgn_psd(freqs, H, psd_exact)

m_exact, _ = train_one(H, N, dt, psd_exact, 'SpectralFilter exact',
                        n_epochs=EPOCHS, device=device)
m_asymp, _ = train_one(H, N, dt, psd_asymp, 'SpectralFilter asymp',
                        n_epochs=EPOCHS, device=device)
m_mlp, _   = train_one(H, N, dt, psd_exact, 'BochnerMLP exact',
                        n_epochs=EPOCHS, device=device, model=BochnerMLP(N))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Generate surrogate variance paths
# ─────────────────────────────────────────────────────────────────────────────

def model_to_V(model: torch.nn.Module, n_paths: int, seed: int = 7) -> np.ndarray:
    """
    Draw fGn increments from trained model, cumsum → fBm path W,
    then V = ξ₀ exp(η W − ½η² Var[W_t]).  Returns (n_paths, N+1).

    Drift uses the EMPIRICAL variance of W across paths (not the theoretical
    t^{2H}) so that E[V_t] = ξ₀ exactly regardless of whether the surrogate's
    cumsum variance matches the reference fBm.  This isolates the spectral
    shape effect (H, correlation structure) from the variance level.
    """
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        dW = model(torch.randn(n_paths, N, device=device)).cpu().numpy()
    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    var_W = W.var(axis=0)                              # (N+1,) cross-sectional
    drift_corr = -0.5 * eta ** 2 * var_W               # E[V_t]=ξ₀ by construction
    return xi0 * np.exp(eta * W + drift_corr[None, :])


print(f"\n=== 3. Generating surrogate variance paths  ({N_SURR:,} each) ===")
V_exact = model_to_V(m_exact, N_SURR)
V_asymp = model_to_V(m_asymp, N_SURR)
V_mlp   = model_to_V(m_mlp,   N_SURR)

for label, V in [('exact', V_exact), ('asymp', V_asymp), ('mlp', V_mlp)]:
    print(f"     E[V_{label}] = {V.mean():.5f}  (target {xi0:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Surrogate IV surfaces
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== 4. Computing surrogate IV surfaces ===")
iv_exact, _, _ = compute_iv_surface(V_exact, dt, rho=0.0,
                                     strikes=_STRIKES_DEFAULT,
                                     maturities=_MATURITIES_DEFAULT)
iv_asymp, _, _ = compute_iv_surface(V_asymp, dt, rho=0.0,
                                     strikes=_STRIKES_DEFAULT,
                                     maturities=_MATURITIES_DEFAULT)
iv_mlp,   _, _ = compute_iv_surface(V_mlp,   dt, rho=0.0,
                                     strikes=_STRIKES_DEFAULT,
                                     maturities=_MATURITIES_DEFAULT)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RMSE table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== 5. IV surface RMSE (vol points) ===")
print(f"{'Model':<28}  {'Overall':>8}  " +
      "  ".join(f"T={Tv:.2f}" for Tv in T_arr))
print("-" * 60)

all_stats = {}
for label, iv_s in [('SpectralFilter (exact PSD)', iv_exact),
                     ('SpectralFilter (asymp PSD)', iv_asymp),
                     ('BochnerMLP     (exact PSD)', iv_mlp)]:
    s = compare_iv_surfaces(iv_ref, iv_s, K_arr, T_arr)
    all_stats[label] = s
    per_mat = "  ".join(f"{r:6.3f}" for r in s['rmse_by_maturity'].values())
    print(f"  {label:<26}  {s['rmse_overall']:8.3f}  {per_mat}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figures
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== 6. Saving figures ===")
plot_iv_comparison(iv_ref, iv_exact, iv_asymp, K_arr, T_arr)
print(f"     IV smile comparison → {FIGURES / 'iv_smile_comparison.png'}")

np.savez(RESULTS / 'surrogate_comparison.npz',
         K_arr=K_arr, T_arr=T_arr,
         iv_ref=iv_ref, iv_exact=iv_exact,
         iv_asymp=iv_asymp, iv_mlp=iv_mlp)
print(f"     Numerical results   → {RESULTS / 'surrogate_comparison.npz'}")
