"""
power_volterra.py  —  V-conditional Volterra with learnable sigma(V) = V^beta.

Extends VolterraSurrogate to handle non-affine rough volatility:
  - beta = 0.5  : Heston-like (sqrt(V))
  - beta = 1.0  : lognormal-rough-vol (V)
  - beta = 0.0  : constant-vol Volterra
  - learnable   : let the data pick.

All parameters (kernel weights w_i, c_drift, c_stoch, theta, beta) are
learnable.  Same autograd-friendly list-based forward pass as
VolterraSurrogate.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma as _gamma


class PowerVolterraSurrogate(nn.Module):
    def __init__(self, N: int, dt: float, H: float = 0.1,
                 V_0: float = 0.04, theta_0: float = 0.04,
                 kappa_0: float = 2.0, nu_0: float = 1.9,
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
        self.theta   = nn.Parameter(torch.tensor(theta_0, dtype=torch.float32))
        self.beta    = nn.Parameter(torch.tensor(beta_0, dtype=torch.float32))
        self.register_buffer('V_0', torch.tensor(V_0, dtype=torch.float32))
        self.N     = N
        self.V_min = V_min

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N) -> V: (B, N+1)."""
        B = z.shape[0]
        V_list = [self.V_0.expand(B)]
        for k in range(self.N):
            V_curr    = torch.stack(V_list, dim=1)              # (B, k+1)
            wk        = self.w[:k + 1].flip(0)
            drift_sum = ((self.theta - V_curr) * wk).sum(dim=1)
            # Generalised diffusion sigma(V) = V^beta  (clamped to >=V_min for stability)
            V_safe    = V_curr.clamp(min=self.V_min)
            V_beta    = V_safe.pow(self.beta)
            stoch_sum = (V_beta * z[:, :k + 1] * wk).sum(dim=1)
            V_next    = (self.V_0
                         + self.c_drift * drift_sum
                         + self.c_stoch * stoch_sum).clamp(min=self.V_min)
            V_list.append(V_next)
        return torch.stack(V_list, dim=1)


# ─── Reference simulator with general sigma(V) for non-affine experiments ────
def simulate_rough_volterra(H: float, kappa: float, theta: float, nu: float,
                            V_0: float, N: int, n_paths: int, T: float = 1.0,
                            beta: float = 0.5, seed: int = 42):
    """
    Forward-Euler Volterra integrator with sigma(V) = nu * V^beta.
    beta = 0.5 reproduces simulate_rough_heston_variance.
    Returns (V, Z, t).
    """
    alpha = H + 0.5
    dt    = T / N
    rng   = np.random.default_rng(seed)
    Z     = rng.standard_normal((n_paths, N))
    i_arr = np.arange(N, dtype=float)
    w     = (i_arr + 1.0) ** (alpha - 1.0)
    c_drift = dt ** alpha       * kappa / _gamma(alpha)
    c_stoch = dt ** (alpha - 0.5) * nu  / _gamma(alpha)

    V = np.zeros((n_paths, N + 1))
    V[:, 0] = V_0
    for k in range(N):
        wk        = w[k::-1]
        drift_sum = (theta - V[:, :k + 1]).dot(wk)
        V_safe    = np.maximum(V[:, :k + 1], 0.0)
        V_beta    = V_safe ** beta
        stoch_sum = (V_beta * Z[:, :k + 1]).dot(wk)
        V[:, k + 1] = np.maximum(V_0 + c_drift * drift_sum + c_stoch * stoch_sum, 0.0)
    return V, Z, np.linspace(0.0, T, N + 1)
