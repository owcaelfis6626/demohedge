"""
implied_vol.py  —  Implied volatility surface for rBergomi surrogate comparison.

Pipeline:
  1. simulate_rbm_stock      : Euler-Maruyama stock paths under rBergomi (ρ=0 default)
  2. mc_call_price           : MC European call price
  3. implied_vol_single      : Black-Scholes IV by Brent root-finding
  4. compute_iv_surface      : full (maturity × strike) IV grid
  5. compare_surfaces        : reference MC truth vs any surrogate, returns RMSE table

Key parameters for the paper experiments (§4.2):
  S0=1, r=0, H=0.1, η=1.9, ξ₀=0.04, ρ=-0.9 (typical rough-vol calibration)
  Strikes  K/S₀ ∈ {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20}
  Maturities T ∈ {0.25, 0.50, 1.00} years
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_rbm_variance

FIGURES = Path(__file__).parent.parent / 'figures'
RESULTS = Path(__file__).parent.parent / 'results'


# ═══════════════════════════════════════════════════════════════════════
# 1.  Stock price simulation under rBergomi
# ═══════════════════════════════════════════════════════════════════════

def simulate_stock(V: np.ndarray, dt: float,
                   innovations: np.ndarray = None,
                   S0: float = 1.0,
                   rho: float = 0.0,
                   seed: int = 42) -> np.ndarray:
    """
    Euler-Maruyama stock paths under stochastic vol model.

        d log S_t = -½ V_t dt + √(V_t dt) dZ_t
        dZ_t = ρ Z_t + √(1-ρ²) Z_t^⊥    (Z_t, Z_t^⊥ ~ iid N(0,1))

    V          : (n_paths, N+1)  instantaneous variance
    innovations: (n_paths, N)    unit-normal BM innovations that drove V;
                                 if provided and rho≠0, used for Cholesky
                                 decomposition to correlate S with V.
                                 Pass None for ρ=0 (independent stock BM).
    Returns S: (n_paths, N+1)
    """
    n_paths, Np1 = V.shape
    N = Np1 - 1
    rng = np.random.default_rng(seed)
    Z_perp = rng.standard_normal((n_paths, N))    # independent BM innovations

    if innovations is not None and rho != 0.0:
        # Cholesky: dZ = ρ Z + √(1-ρ²) Z^⊥, where Z = innovations ~ N(0,1)
        Z_v = innovations / (np.std(innovations) + 1e-12)   # normalise to unit variance
        dZ  = rho * Z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * Z_perp
    else:
        dZ = Z_perp

    # Euler-Maruyama: use variance at left endpoint of each interval
    V_left = V[:, :-1].clip(0)
    log_increments = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ
    log_S = np.zeros((n_paths, N + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)
    return np.exp(log_S)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Black-Scholes analytics
# ═══════════════════════════════════════════════════════════════════════

def bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes call price (σ > 0, T > 0)."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_single(price: float, S0: float, K: float,
                        r: float, T: float,
                        lo: float = 1e-4, hi: float = 10.0) -> float:
    """
    Implied vol by Brent root-finding on the Black-Scholes call.
    Returns np.nan if the price is outside the no-arbitrage bounds.
    """
    intrinsic = max(S0 * np.exp(-r * 0) - K * np.exp(-r * T), 0.0)
    upper     = S0
    if not (intrinsic < price < upper):
        return np.nan
    try:
        return brentq(lambda s: bs_call(S0, K, r, s, T) - price, lo, hi,
                      xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════
# 3.  IV surface computation
# ═══════════════════════════════════════════════════════════════════════

_STRIKES_DEFAULT    = np.array([0.80, 0.85, 0.90, 0.95, 1.00,
                                  1.05, 1.10, 1.15, 1.20])
_MATURITIES_DEFAULT = np.array([0.25, 0.50, 1.00])


def compute_iv_surface(V: np.ndarray, dt: float,
                        innovations: np.ndarray = None,
                        S0: float = 1.0,
                        r: float = 0.0,
                        rho: float = 0.0,
                        strikes: np.ndarray = _STRIKES_DEFAULT,
                        maturities: np.ndarray = _MATURITIES_DEFAULT,
                        seed: int = 42) -> tuple:
    """
    Compute the implied volatility surface from variance paths V.

    Returns:
        iv   : (n_maturities, n_strikes)  — IV in decimal (0.20 = 20%)
        K_arr: (n_strikes,)
        T_arr: (n_maturities,)
    """
    S = simulate_stock(V, dt, innovations=innovations, S0=S0, rho=rho, seed=seed)
    n_mat, n_str = len(maturities), len(strikes)
    iv = np.full((n_mat, n_str), np.nan)
    K_arr = strikes * S0

    for i, T in enumerate(maturities):
        idx = min(int(round(T / dt)), V.shape[1] - 1)
        S_T = S[:, idx]
        for j, K in enumerate(K_arr):
            price = np.mean(np.maximum(S_T - K, 0.0)) * np.exp(-r * T)
            iv[i, j] = implied_vol_single(price, S0, K, r, T)

    return iv, K_arr, maturities


# ═══════════════════════════════════════════════════════════════════════
# 3b.  Delta surface  (bump-and-reprice with frozen paths)
# ═══════════════════════════════════════════════════════════════════════

def compute_delta_surface(V: np.ndarray, innovations: np.ndarray,
                           dt: float, rho: float,
                           S0: float = 1.0, r: float = 0.0,
                           bump: float = 0.01,
                           strikes: np.ndarray = _STRIKES_DEFAULT,
                           maturities: np.ndarray = _MATURITIES_DEFAULT,
                           seed: int = 42) -> np.ndarray:
    """
    Finite-difference delta surface using frozen paths (common random numbers).

    For each (T, K): Δ ≈ [Price(S₀+ε) − Price(S₀−ε)] / (2ε)
    with V and the stock BM innovations held fixed across bumps.

    V          : (n_paths, N+1)  variance paths
    innovations: (n_paths, N)    unit-normal BM innovations that drove V
    bump       : ε as a fraction of S0  (default 1%)

    Returns delta: (n_maturities, n_strikes)
    """
    eps = bump * S0
    # Generate stock BM innovations once, reuse for all three prices
    rng   = np.random.default_rng(seed)
    n_paths, N = innovations.shape
    Z_perp = rng.standard_normal((n_paths, N))

    def _price(S0_val: float) -> np.ndarray:
        """(n_maturities, n_strikes) call prices for a given S₀."""
        Z_v = innovations / (np.std(innovations) + 1e-12)
        dZ  = rho * Z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * Z_perp
        V_left = V[:, :-1].clip(0)
        log_inc = -0.5 * V_left * dt + np.sqrt(V_left * dt) * dZ
        log_S = np.log(S0_val) + np.concatenate(
            [np.zeros((n_paths, 1)),
             np.cumsum(log_inc, axis=1)], axis=1)         # (n_paths, N+1)
        S = np.exp(log_S)
        K_arr = strikes * S0
        prices = np.full((len(maturities), len(strikes)), np.nan)
        for i, T in enumerate(maturities):
            idx  = min(int(round(T / dt)), N)
            S_T  = S[:, idx]
            for j, K in enumerate(K_arr):
                prices[i, j] = np.mean(np.maximum(S_T - K, 0.0)) * np.exp(-r * T)
        return prices

    p_up   = _price(S0 + eps)
    p_down = _price(S0 - eps)
    delta  = (p_up - p_down) / (2.0 * eps)
    return delta


# ═══════════════════════════════════════════════════════════════════════
# 4.  Surface comparison  (reference MC vs surrogate)
# ═══════════════════════════════════════════════════════════════════════

def compare_iv_surfaces(iv_ref: np.ndarray,
                         iv_surr: np.ndarray,
                         K_arr: np.ndarray,
                         T_arr: np.ndarray,
                         S0: float = 1.0) -> dict:
    """
    Compute RMSE table and summary statistics.
    iv_ref, iv_surr: (n_mat, n_str)  — in decimal vol units.
    Returns dict with per-maturity RMSE (in vol points = 1e-2) and overall RMSE.
    """
    diff = (iv_surr - iv_ref) * 100      # vol points
    valid = np.isfinite(diff)
    rmse_overall = np.sqrt(np.mean(diff[valid] ** 2))
    rmse_by_mat  = [np.sqrt(np.mean(diff[i, valid[i]] ** 2))
                    for i in range(len(T_arr))]
    return dict(rmse_overall=rmse_overall,
                rmse_by_maturity=dict(zip(T_arr, rmse_by_mat)),
                diff_vp=diff)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Figures
# ═══════════════════════════════════════════════════════════════════════

def plot_iv_surface(iv: np.ndarray, K_arr: np.ndarray, T_arr: np.ndarray,
                    S0: float = 1.0, title: str = 'IV surface',
                    fname: str = 'iv_surface.png') -> None:
    moneyness = K_arr / S0
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, T in enumerate(T_arr):
        mask = np.isfinite(iv[i])
        ax.plot(moneyness[mask], iv[i][mask] * 100,
                marker='o', ms=4, label=f'T={T:.2f}y')
    ax.set_xlabel('Moneyness K/S₀')
    ax.set_ylabel('Implied vol (%)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=120)
    plt.close(fig)


def plot_iv_comparison(iv_ref, iv_bochner, iv_asymp,
                        K_arr, T_arr, S0=1.0) -> None:
    """Side-by-side IV smile comparison for each maturity."""
    n_mat = len(T_arr)
    moneyness = K_arr / S0
    fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 4), sharey=True)
    if n_mat == 1:
        axes = [axes]
    for i, (ax, T) in enumerate(zip(axes, T_arr)):
        valid_r = np.isfinite(iv_ref[i])
        valid_b = np.isfinite(iv_bochner[i])
        valid_a = np.isfinite(iv_asymp[i])
        ax.plot(moneyness[valid_r], iv_ref[i][valid_r] * 100,
                'k-o', ms=4, lw=2, label='MC truth')
        ax.plot(moneyness[valid_b], iv_bochner[i][valid_b] * 100,
                'r--s', ms=4, lw=1.5, label='Bochner PINN (exact PSD)')
        ax.plot(moneyness[valid_a], iv_asymp[i][valid_a] * 100,
                'm:^', ms=4, lw=1.5, label='Asymptotic PSD')
        ax.set_xlabel('Moneyness K/S₀')
        ax.set_title(f'T = {T:.2f} yr')
        if i == 0:
            ax.set_ylabel('Implied vol (%)')
        ax.legend(fontsize=7)
    fig.suptitle('rBergomi implied volatility smile comparison', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / 'iv_smile_comparison.png', dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Main: reference surface
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    H, eta, xi0 = 0.1, 1.9, 0.04
    N, T = 252, 1.0
    dt = T / N
    n_paths_ref = 5000

    # ρ=0: using fBm increments as stock correlation driver violates the
    # martingale property (fGn has E[dW^H_k|F_{k-1}]≠0 for H≠0.5), causing
    # E[S_T] > 1 and inflated IVs.  The correct ρ≠0 simulation requires the
    # hybrid scheme (Bennedsen-Lunde-Pakkanen 2017) to share standard-BM
    # innovations between V and S.  For this experiment we use ρ=0.
    print(f"Computing reference IV surface (H={H}, n_paths={n_paths_ref}, rho=0)...")
    V_ref, _, t = simulate_rbm_variance(H, eta, xi0, N, n_paths_ref, T, seed=1)

    iv_ref, K_arr, T_arr = compute_iv_surface(
        V_ref, dt, innovations=None, rho=0.0,
        strikes=_STRIKES_DEFAULT, maturities=_MATURITIES_DEFAULT)

    print("\nReference IV surface (%):")
    header = "  T\\K  " + "  ".join(f"{k:.2f}" for k in _STRIKES_DEFAULT)
    print(header)
    for i, T_val in enumerate(_MATURITIES_DEFAULT):
        row = f"  {T_val:.2f} " + "  ".join(
            f"{v*100:5.2f}" if np.isfinite(v) else "   nan" for v in iv_ref[i])
        print(row)

    plot_iv_surface(iv_ref, K_arr, T_arr, title='Reference rBergomi IV surface',
                    fname='iv_surface_reference.png')
    np.savez(RESULTS / 'iv_reference.npz',
             iv=iv_ref, K_arr=K_arr, T_arr=T_arr)
    print(f"\n  Figure → {FIGURES/'iv_surface_reference.png'}")
