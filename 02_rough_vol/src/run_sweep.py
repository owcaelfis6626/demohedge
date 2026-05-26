"""
run_sweep.py  —  Run surrogate comparison for a single (H, eta) configuration.
Usage:  python run_sweep.py  H  eta  tag
  e.g.  python run_sweep.py  0.05  1.9  H005
Output: results/sweep_<tag>.npz  +  stdout table
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_variance
from bochner_pinn import (SpectralFilter, BochnerMLP, get_exact_fgn_psd,
                           asymptotic_fgn_psd, train_one)
from implied_vol import (compute_iv_surface, compare_iv_surfaces,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT,
                          RESULTS)

H    = float(sys.argv[1])
eta  = float(sys.argv[2])
tag  = sys.argv[3]
xi0  = 0.04
N, T = 252, 1.0
dt   = T / N
N_REF  = 10_000
N_SURR =  5_000
EPOCHS = 3_000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n=== H={H}  eta={eta}  tag={tag}  device={device} ===\n")

# 1. Reference
V_ref, _, _ = simulate_rbm_variance(H, eta, xi0, N, N_REF, T, seed=1)
iv_ref, K_arr, T_arr = compute_iv_surface(V_ref, dt, rho=0.0,
                                           strikes=_STRIKES_DEFAULT,
                                           maturities=_MATURITIES_DEFAULT)

# 2. Train
freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
psd_asymp = asymptotic_fgn_psd(freqs, H, psd_exact)

m_exact, _ = train_one(H, N, dt, psd_exact, f'SpectralFilter exact [{tag}]',
                        n_epochs=EPOCHS, device=device)
m_asymp, _ = train_one(H, N, dt, psd_asymp, f'SpectralFilter asymp [{tag}]',
                        n_epochs=EPOCHS, device=device)
m_mlp, _   = train_one(H, N, dt, psd_exact, f'BochnerMLP exact [{tag}]',
                        n_epochs=EPOCHS, device=device, model=BochnerMLP(N))

# 3. Generate surrogate paths
def model_to_V(model):
    torch.manual_seed(7)
    model.eval()
    with torch.no_grad():
        dW = model(torch.randn(N_SURR, N, device=device)).cpu().numpy()
    W = np.zeros((N_SURR, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    var_W = W.var(axis=0)
    drift_corr = -0.5 * eta ** 2 * var_W
    return xi0 * np.exp(eta * W + drift_corr[None, :])

V_exact = model_to_V(m_exact)
V_asymp = model_to_V(m_asymp)
V_mlp   = model_to_V(m_mlp)

# 4. IV surfaces
iv_exact, _, _ = compute_iv_surface(V_exact, dt, rho=0.0,
                                     strikes=_STRIKES_DEFAULT,
                                     maturities=_MATURITIES_DEFAULT)
iv_asymp, _, _ = compute_iv_surface(V_asymp, dt, rho=0.0,
                                     strikes=_STRIKES_DEFAULT,
                                     maturities=_MATURITIES_DEFAULT)
iv_mlp,   _, _ = compute_iv_surface(V_mlp,   dt, rho=0.0,
                                     strikes=_STRIKES_DEFAULT,
                                     maturities=_MATURITIES_DEFAULT)

# 5. RMSE table
print(f"\n{'Model':<28}  {'Overall':>8}  " +
      "  ".join(f"T={Tv:.2f}" for Tv in T_arr))
print("-" * 60)
for label, iv_s in [('SpectralFilter (exact PSD)', iv_exact),
                     ('SpectralFilter (asymp PSD)', iv_asymp),
                     ('BochnerMLP     (exact PSD)', iv_mlp)]:
    s = compare_iv_surfaces(iv_ref, iv_s, K_arr, T_arr)
    per_mat = "  ".join(f"{r:6.3f}" for r in s['rmse_by_maturity'].values())
    print(f"  {label:<26}  {s['rmse_overall']:8.3f}  {per_mat}")

# 6. Save
np.savez(RESULTS / f'sweep_{tag}.npz',
         K_arr=K_arr, T_arr=T_arr,
         iv_ref=iv_ref, iv_exact=iv_exact,
         iv_asymp=iv_asymp, iv_mlp=iv_mlp,
         H=H, eta=eta)
print(f"\nSaved → results/sweep_{tag}.npz")
