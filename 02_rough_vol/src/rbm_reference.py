"""
rBergomi reference solver and spectral diagnostics.

Priority one from the plan: simulate the rough Bergomi variance process,
compute its power spectral density, verify the power-law slope matches
-(2H+1), and estimate H from the log-log PSD.

Simulation method: Wood-Chan circulant embedding (exact, O(N log N)).
"""

import numpy as np
from numpy.fft import fft, ifft, rfft, rfftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# fBm simulation: Wood-Chan circulant embedding (Dietrich & Newsam 1997)
# Exact simulation of stationary-increment Gaussian process on a grid.
# ---------------------------------------------------------------------------

def _fbm_covariance_row(H: float, N: int) -> np.ndarray:
    """
    First row of the covariance matrix of fBm increments.
    C[j] = Cov(B^H_1 - B^H_0, B^H_{j+1} - B^H_j)
           = 0.5 * (|j+1|^{2H} + |j-1|^{2H} - 2|j|^{2H})
    for j = 0, 1, ..., N-1.
    """
    j = np.arange(N, dtype=float)
    c = 0.5 * (np.abs(j + 1) ** (2 * H) + np.abs(j - 1) ** (2 * H)
               - 2 * np.abs(j) ** (2 * H))
    return c


def simulate_fbm_increments(H: float, N: int, n_paths: int,
                             seed: int = 42) -> np.ndarray:
    """
    Simulate n_paths independent fBm increment sequences of length N
    using circulant embedding.

    Returns array of shape (n_paths, N) where each row is the sequence
    of N increments dB^H_k = B^H_{k/N} - B^H_{(k-1)/N}.

    Time is rescaled so B^H_1 has unit variance (Var = T^{2H} with T=1).
    """
    rng = np.random.default_rng(seed)

    # Covariance of unit-time increments
    c = _fbm_covariance_row(H, N)

    # Embed into circulant of size 2N (guaranteed positive semi-definite
    # for the standard fBm covariance)
    M = 2 * N
    row = np.zeros(M)
    row[:N] = c
    row[N] = 0.0
    row[N + 1:] = c[N - 1:0:-1]

    # Eigenvalues of the circulant via FFT
    lam = np.real(fft(row))
    if np.any(lam < -1e-10):
        raise RuntimeError(f"Circulant not PSD; min eigenvalue {lam.min():.3e}")
    lam = np.maximum(lam, 0.0)
    # Use sqrt(lam) — not sqrt(lam/M) — so that Var(increment) = c[0] = 1.
    # Proof: W_n = Re(IFFT(sqrt(lam)*FFT(Z)))_n = Σ_j Z_j h_{n-j} where h=IFFT(sqrt(lam)).
    # Var(W_n) = ||h||² = (1/M) Σ_k lam_k = c[0] = 1.
    sqrt_lam = np.sqrt(lam)

    # Generate complex Gaussian (CN(0,2)) array
    Z = rng.standard_normal((n_paths, M)) + 1j * rng.standard_normal((n_paths, M))
    W = np.real(ifft(sqrt_lam[None, :] * fft(Z, axis=1), axis=1))

    # Extract the first N values: these are the fBm increments
    increments = W[:, :N]  # (n_paths, N)
    return increments


def simulate_fbm_paths(H: float, N: int, n_paths: int,
                       T: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Simulate fBm paths B^H on [0, T] with N steps.
    Returns (n_paths, N+1); first column is zero.
    """
    dt = T / N
    increments = simulate_fbm_increments(H, N, n_paths, seed)
    # Rescale: Var(B^H_{dt}) = dt^{2H}
    increments *= dt ** H
    paths = np.zeros((n_paths, N + 1))
    paths[:, 1:] = np.cumsum(increments, axis=1)
    return paths


def simulate_rbm_variance(H: float, eta: float, xi0: float,
                          N: int, n_paths: int,
                          T: float = 1.0, seed: int = 42) -> tuple:
    """
    Rough Bergomi variance process:
        V_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})

    where W^H_t is the Riemann-Liouville fBm.  We approximate W^H by
    standard fBm here (they share the same spectral properties for our
    verification purposes; see plan note on Volterra kernel).

    Returns:
        V  : (n_paths, N+1)  — instantaneous variance
        W  : (n_paths, N+1)  — fBm driving process
        t  : (N+1,)          — time grid
    """
    t = np.linspace(0.0, T, N + 1)
    W = simulate_fbm_paths(H, N, n_paths, T, seed)
    drift = -0.5 * eta ** 2 * t[None, :] ** (2 * H)
    V = xi0 * np.exp(eta * W + drift)
    return V, W, t


# ---------------------------------------------------------------------------
# Power spectral density estimation
# ---------------------------------------------------------------------------

def compute_psd(X: np.ndarray, dt: float) -> tuple:
    """
    Estimate the power spectral density of a 2D array X (n_paths, N)
    by averaging periodograms across paths.

    Returns (freqs, psd_mean) excluding the DC component.
    """
    n_paths, N = X.shape
    freqs = rfftfreq(N, d=dt)[1:]   # exclude f=0

    psds = []
    for i in range(n_paths):
        x = X[i] - X[i].mean()      # demean
        F = rfft(x)
        # one-sided periodogram, normalised to power per unit frequency
        P = (2.0 * dt / N) * np.abs(F[1:]) ** 2
        psds.append(P)

    return freqs, np.mean(psds, axis=0)


def estimate_hurst_structure_function(W: np.ndarray, dt: float,
                                      lags: tuple = (1, 2, 3, 5, 8, 13, 21, 34, 55)
                                      ) -> dict:
    """
    Estimate H from the second-order structure function (variogram).

    For fBm:  E[(W_{t+τ} - W_t)²] = (τ·dt)^{2H}
    so       log E[|ΔW_τ|²] = 2H·log(τ·dt) + const
    and       H = slope / 2  (OLS on log-log).

    Advantages over PSD: unbiased for all H (not just asymptotic f→0),
    robust to the non-stationarity of fBm levels.

    W: (n_paths, N+1) — fBm paths (W[:,0] = 0).
    """
    n_paths, Np1 = W.shape
    N = Np1 - 1
    sf_vals, tau_vals = [], []
    for lag in lags:
        if lag >= N:
            break
        diff = W[:, lag:] - W[:, :-lag]
        sf_vals.append(np.mean(diff ** 2))
        tau_vals.append(lag * dt)

    log_tau = np.log(tau_vals)
    log_sf  = np.log(sf_vals)
    slope, intercept = np.polyfit(log_tau, log_sf, 1)
    H_est = slope / 2.0
    return dict(H_est=H_est, slope=slope, intercept=intercept,
                tau_vals=tau_vals, sf_vals=sf_vals)


def estimate_hurst_from_psd(freqs: np.ndarray, psd: np.ndarray,
                             f_min_frac: float = 0.02,
                             f_max_frac: float = 0.30,
                             use_fgn: bool = True) -> dict:
    """
    Estimate the Hurst exponent from the slope of the log-log PSD.

    Two conventions depending on what X is:

    use_fgn=True  (default, recommended):
      X = fBm increments (fractional Gaussian noise, stationary).
      S(f) ~ C * |f|^{1-2H}  →  slope = 1-2H  →  H = (1 - slope) / 2.
      For H=0.1: expected slope = +0.8 (PSD increases with freq — rough).
      For H=0.5: expected slope = 0    (white noise — BM increments).

    use_fgn=False:
      X = fBm levels (non-stationary; use only after drift removal).
      S(f) ~ C * |f|^{-(2H+1)}  →  slope = -(2H+1)  →  H = -(slope+1) / 2.

    f_min_frac, f_max_frac: fitting window as fraction of Nyquist.
    """
    f_nyq = freqs.max()
    mask = (freqs >= f_nyq * f_min_frac) & (freqs <= f_nyq * f_max_frac)
    if mask.sum() < 10:
        raise ValueError("Too few frequencies in fitting window.")

    log_f   = np.log(freqs[mask])
    log_psd = np.log(psd[mask])

    A = np.column_stack([log_f, np.ones_like(log_f)])
    coef, _, _, _ = np.linalg.lstsq(A, log_psd, rcond=None)
    slope, intercept = coef

    if use_fgn:
        H_est = (1.0 - slope) / 2.0   # S ~ f^{1-2H}
    else:
        H_est = -(slope + 1.0) / 2.0  # S ~ f^{-(2H+1)}

    return dict(H_est=H_est, slope=slope, intercept=intercept,
                freqs_fit=freqs[mask], psd_fit=psd[mask],
                use_fgn=use_fgn)


# ---------------------------------------------------------------------------
# Rough Heston variance process  (Volterra Euler discretisation)
# ---------------------------------------------------------------------------

def simulate_rough_heston_variance(H: float, kappa: float, theta: float,
                                    nu: float, V0: float,
                                    N: int, n_paths: int,
                                    T: float = 1.0,
                                    seed: int = 42) -> tuple:
    """
    Rough Heston variance via a forward-Euler Volterra scheme.

        V_t = V_0 + (1/Γ(α)) ∫_0^t (t-s)^{α-1} [κ(θ-V_s) ds + ν√V_s dB_s]

    α = H + 1/2  ∈ (1/2, 1)  (rough regime, H ∈ (0,1/2))

    Discretisation on uniform grid t_k = k·dt, dt = T/N:

        V_{k+1} = V_0
                  + (dt^α κ / Γ(α))   · Σ_{j=0}^{k} (k+1-j)^{α-1} (θ - V_j)
                  + (dt^{α-½} ν / Γ(α))· Σ_{j=0}^{k} (k+1-j)^{α-1} √V_j Z_j

    where Z_j ~ iid N(0,1).  The dot product at each step is vectorised over
    all paths simultaneously: cost O(N² · n_paths) total.  For N=252, n_paths=
    10k: ≈ 300M multiply-adds, ~0.5 s on CPU.

    Returns
    -------
    V : (n_paths, N+1)   instantaneous variance (clamped ≥ 0)
    Z : (n_paths, N)     standard BM innovations driving V; used for
                         the S-V correlation dZ = ρ Z + √(1-ρ²) Z^⊥
    t : (N+1,)           time grid
    """
    from scipy.special import gamma as _gamma

    alpha = H + 0.5
    dt    = T / N
    rng   = np.random.default_rng(seed)
    Z     = rng.standard_normal((n_paths, N))         # BM innovations

    # Kernel weights: w[i] = (i+1)^{α-1} for i = 0..N-1  (1-indexed lag l = i+1)
    i_arr = np.arange(N, dtype=float)
    w     = (i_arr + 1.0) ** (alpha - 1.0)            # shape (N,)

    c_drift = dt ** alpha       * kappa / _gamma(alpha)
    c_stoch = dt ** (alpha - 0.5) * nu  / _gamma(alpha)

    V = np.zeros((n_paths, N + 1))
    V[:, 0] = V0

    for k in range(N):
        # Kernel weights for step k: w[k-j] for j=0..k  (lag l = k+1-j ∈ {1..k+1})
        wk = w[k::-1]                                  # shape (k+1,)

        drift_sum = (theta - V[:, :k+1]).dot(wk)       # (n_paths,)
        sqV       = np.sqrt(np.maximum(V[:, :k+1], 0.0))
        stoch_sum = (sqV * Z[:, :k+1]).dot(wk)         # (n_paths,)

        V[:, k+1] = np.maximum(V0 + c_drift * drift_sum + c_stoch * stoch_sum, 0.0)

    t = np.linspace(0.0, T, N + 1)
    return V, Z, t


def simulate_rbm_blp(H: float, eta: float, xi0: float,
                     N: int, n_paths: int,
                     T: float = 1.0, rho: float = 0.0,
                     seed: int = 42) -> tuple:
    """
    rBergomi via Riemann-Liouville discretisation (BLP hybrid-scheme style).

    Correctly implements rho != 0 by correlating the stock BM with the
    DRIVING standard BM Z that builds W^H — not with the fGn increments.

        W^H_{kh} = (dt^{H+1/2} / Gamma(H+1/2))
                   * sum_{j=0}^{k-1} (k-j)^{H-1/2} * Z_j

        V_t = xi0 * exp(eta * W^H_t  -  1/2 * eta^2 * Var[W^H_t])

        dlog S_k = -1/2 V_k dt + sqrt(V_k dt) * (rho*Z_k + sqrt(1-rho^2)*Z_perp_k)

    Vectorised as a single matrix multiply  Z @ M.T  where M is the (N x N)
    lower-triangular Toeplitz kernel matrix.  O(N^2 * n_paths) flops; for
    N=252, n_paths=8k this is ~0.5 s on a modern CPU.

    Returns
    -------
    V    : (n_paths, N+1)  variance paths
    dZ_S : (n_paths, N)    pre-correlated stock BM innovations  N(0,1)
    t    : (N+1,)          time grid
    """
    from scipy.special import gamma as _gamma
    from scipy.linalg import toeplitz as _toeplitz

    dt = T / N
    rng    = np.random.default_rng(seed)
    Z      = rng.standard_normal((n_paths, N))   # driving BM for W^H
    Z_perp = rng.standard_normal((n_paths, N))   # independent stock BM

    # RL kernel: g[i] = (i+1)^{H-1/2} / Gamma(H+1/2)  for i = 0..N-1
    i_arr = np.arange(1, N + 1, dtype=float)
    g = i_arr ** (H - 0.5) / _gamma(H + 0.5)    # (N,)

    # Lower-triangular Toeplitz M: M[k, j] = g[k-j] for 0 <= j <= k
    first_col = g
    first_row = np.r_[g[0], np.zeros(N - 1)]
    M = np.tril(_toeplitz(first_col, first_row))  # (N, N)

    # W^H[:, 1:] = dt^H * (Z @ M.T)   shape (n_paths, N)
    W_H = np.zeros((n_paths, N + 1))
    W_H[:, 1:] = dt ** H * (Z @ M.T)

    # Martingale correction via empirical cross-sectional variance
    var_W = W_H.var(axis=0)                      # (N+1,)
    V = xi0 * np.exp(eta * W_H - 0.5 * eta ** 2 * var_W[None, :])

    # Correlated stock BM innovations
    dZ_S = rho * Z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * Z_perp  # (n_paths, N)

    return V, dZ_S, np.linspace(0.0, T, N + 1)


# ---------------------------------------------------------------------------
# Verification run
# ---------------------------------------------------------------------------

def run_verification(H: float = 0.1, eta: float = 1.9, xi0: float = 0.04,
                     N: int = 252, n_paths: int = 2000, T: float = 1.0,
                     seed: int = 0) -> dict:
    """
    End-to-end verification:
    1. Simulate rBergomi variance paths (and the underlying fBm W^H)
    2. Compute PSD of fBm INCREMENTS dW (stationary — avoids non-stationarity bias)
       S_fGn(f) ~ C |f|^{1-2H}  →  slope = 1-2H  →  H = (1-slope)/2
    3. Estimate H from log-log slope
    4. Save diagnostic figures

    Note on approach: fBm W^H is non-stationary; computing the periodogram of
    its levels gives a biased H estimate. The increments (fractional Gaussian
    noise, fGn) ARE stationary and their PSD has a clean power-law slope 1-2H.
    For H=0.1 the expected slope is +0.8 (PSD increases with frequency — rough).
    """
    print(f"Simulating rBergomi: H={H}, eta={eta}, xi0={xi0}, "
          f"N={N}, n_paths={n_paths}")

    V, W, t = simulate_rbm_variance(H, eta, xi0, N, n_paths, T, seed)
    log_V = np.log(V[:, 1:])   # shape (n_paths, N), skip t=0

    dt = T / N

    # Primary H estimator: structure function E[(W_{t+τ}-W_t)²] = (τ dt)^{2H}.
    # Unbiased for all H; does not suffer from mid-frequency PSD deviation.
    sf_result = estimate_hurst_structure_function(W, dt)
    H_est = sf_result['H_est']

    # Secondary diagnostic: PSD of fBm increments dW (stationary fGn).
    # Note: asymptotic S(f) ~ f^{1-2H} only holds for f ≪ f_nyq.
    # In the mid-frequency window the slope is steeper; this biases OLS H_est.
    dW = np.diff(W, axis=1)    # W has shape (n_paths, N+1)
    freqs, psd = compute_psd(dW, dt)
    expected_slope = 1.0 - 2.0 * H
    psd_result = estimate_hurst_from_psd(freqs, psd, use_fgn=True)
    psd_slope = psd_result['slope']
    H_psd = psd_result['H_est']

    print(f"  True H             : {H:.4f}")
    print(f"  H_est (struct.fn.) : {H_est:.4f}  (error {H_est-H:+.4f})")
    print(f"  H_est (PSD OLS)    : {H_psd:.4f}  (error {H_psd-H:+.4f})  "
          f"[slope {psd_slope:.3f}, expected {expected_slope:.3f}; "
          f"mid-freq bias expected for H<0.5]")

    # --- Figure 1: sample variance paths ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(min(20, n_paths)):
        axes[0].plot(t, V[i], alpha=0.3, lw=0.7, color='steelblue')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('V_t (instantaneous variance)')
    axes[0].set_title(f'rBergomi variance paths  (H={H}, eta={eta})')

    for i in range(min(20, n_paths)):
        axes[1].plot(t[1:], log_V[i], alpha=0.3, lw=0.7, color='darkorange')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('log V_t')
    axes[1].set_title('Log-variance paths')

    fig.tight_layout()
    fig.savefig(FIGURES / f'rbm_paths_H{H:.2f}.png', dpi=120)
    plt.close(fig)

    # --- Figure 2: PSD of fBm increments (fGn) — log-log ---
    # Expected: S(f) ~ f^{1-2H}.  For H=0.1: slope = +0.8 (increasing).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(freqs, psd, alpha=0.8, lw=1.2, color='steelblue',
              label='Empirical PSD of fBm increments dW (mean over paths)')

    # Theoretical power law: S(f) = C * f^{1-2H}
    C_theory = np.exp(psd_result['intercept'])
    psd_theory = C_theory * freqs ** expected_slope
    ax.loglog(freqs, psd_theory, 'r--', lw=1.5,
              label=f'Power law  f^{{1-2H}} = f^{{{expected_slope:.2f}}}  (true H={H})')

    # OLS fitted line
    psd_fit_line = np.exp(psd_result['intercept']) * freqs ** psd_slope
    ax.loglog(freqs, psd_fit_line, 'g:', lw=1.5,
              label=f'OLS fit  slope={psd_slope:.3f}  → H_psd={H_psd:.3f}  '
                    f'[struct.fn. H={H_est:.3f}]')

    ax.set_xlabel('Frequency (cycles / year)')
    ax.set_ylabel('Power spectral density')
    ax.set_title(f'fGn PSD (fBm increments) — rBergomi (H={H}, n_paths={n_paths})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f'rbm_psd_H{H:.2f}.png', dpi=120)
    plt.close(fig)

    print(f"  Figures saved to {FIGURES}")

    return dict(H_true=H, H_est=H_est, H_psd=H_psd, psd_slope=psd_slope,
                freqs=freqs, psd=psd, V=V, W=W, t=t)


def sweep_hurst(H_values=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
                N: int = 252, n_paths: int = 2000) -> None:
    """
    Run verification across several H values and summarise estimation errors.
    """
    print("\n=== Hurst sweep (structure function estimator) ===")
    print(f"{'H_true':>8}  {'H_sf':>8}  {'err_sf':>8}  {'H_psd':>8}  "
          f"{'err_psd':>8}  {'psd_slope':>10}  {'exp_slope':>10}")
    print("-" * 75)
    for H in H_values:
        res = run_verification(H=H, N=N, n_paths=n_paths, seed=42)
        H_est = res['H_est']
        H_psd = res['H_psd']
        psd_slope = res['psd_slope']
        print(f"{H:>8.3f}  {H_est:>8.3f}  {H_est-H:>+8.4f}  "
              f"{H_psd:>8.3f}  {H_psd-H:>+8.4f}  "
              f"{psd_slope:>10.3f}  {1-2*H:>10.3f}")

    # Summary figure
    H_values = list(H_values)
    H_ests = []
    for H in H_values:
        res = run_verification(H=H, N=N, n_paths=n_paths, seed=42)
        H_ests.append(res['H_est'])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(H_values, H_values, 'k--', lw=1, label='Perfect recovery')
    ax.scatter(H_values, H_ests, zorder=3, color='steelblue',
               label='Structure function estimate')
    ax.set_xlabel('True H')
    ax.set_ylabel('Estimated H')
    ax.set_title('Hurst exponent recovery (structure function)\n'
                 f'(rBergomi, N={N}, n_paths={n_paths})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'hurst_recovery.png', dpi=120)
    plt.close(fig)
    print(f"\nSummary figure saved to {FIGURES / 'hurst_recovery.png'}")


if __name__ == '__main__':
    # Primary verification at the financially relevant H = 0.1
    run_verification(H=0.1, eta=1.9, xi0=0.04, N=252, n_paths=2000)
    run_verification(H=0.1, eta=1.9, xi0=0.04, N=1000, n_paths=2000)

    # Sweep across H values to confirm estimator is well-calibrated
    sweep_hurst(n_paths=2000)
