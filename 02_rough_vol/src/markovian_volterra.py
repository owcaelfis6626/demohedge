"""
markovian_volterra.py  —  sub-quadratic Volterra surrogate.

Replace the O(N^2) Volterra sum with M parallel exponential states.
RL kernel  w[i] = (i+1)^{alpha-1} / Gamma(alpha)  is approximated by
   w[i] ≈ sum_{m=1..M} a_m * r_m^i,   r_m = exp(-lambda_m)
giving the recursion
   S_m(k) = X_k + r_m * S_m(k-1)
which costs O(M) per time step (vs O(N) for the full sum).
For M = 5..10, this is sub-quadratic per path.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma as _gamma
from scipy.optimize import nnls


def fit_rl_exponentials(N: int, H: float, M: int = 5, seed: int = 0):
    """
    Fit M nonneg-weighted exponentials to RL kernel w[i] = (i+1)^{H-1/2}/Gamma(H+1/2),
    i = 0..N-1.  Uses geometrically spaced rates plus NNLS for weights.

    Returns (a, lam) both length M, both float64 numpy.
    """
    alpha   = H + 0.5
    i_arr   = np.arange(N, dtype=np.float64)
    w_true  = (i_arr + 1.0) ** (alpha - 1.0) / _gamma(alpha)

    # Geometric rate grid covering [1/N, 1]
    lam = np.geomspace(1.0 / N, 1.0, M)
    # Design matrix: A[i, m] = exp(-lam_m * i)
    A = np.exp(-np.outer(i_arr, lam))
    # NNLS for nonneg weights
    a, _ = nnls(A, w_true)
    # Diagnostic fit error
    w_fit = A @ a
    rel_err = np.linalg.norm(w_true - w_fit) / np.linalg.norm(w_true)
    return a, lam, rel_err


class MarkovianVolterraSurrogate(nn.Module):
    """
    Sub-quadratic V-conditional Volterra surrogate.

        V_{k+1} = V_0 + c_drift * sum_m a_m * S_drift_m(k)
                       + c_stoch * sum_m a_m * S_stoch_m(k)
        S_drift_m(k) = (theta - V_k) + r_m * S_drift_m(k-1)
        S_stoch_m(k) = sqrt(V_k) * z_k + r_m * S_stoch_m(k-1)

    Learnable: a (M,), lam (M,), c_drift, c_stoch, theta.
    Init from fit_rl_exponentials at rough-Heston scalars.
    """

    def __init__(self, N: int, dt: float, H: float = 0.1, M: int = 5,
                 V_0: float = 0.04, theta_0: float = 0.04,
                 kappa_0: float = 2.0, nu_0: float = 1.9,
                 V_min: float = 1.0e-6):
        super().__init__()
        alpha = H + 0.5

        # Fit RL kernel with M exponentials
        a_np, lam_np, fit_err = fit_rl_exponentials(N, H, M=M)
        self.fit_rel_err = fit_err

        # The c_drift, c_stoch scalars absorb dt^alpha and dt^{alpha-1/2}
        # (consistent with VolterraSurrogate).  We additionally absorb the
        # 1/Gamma(alpha) into the fitted a (it's already in w_true above),
        # so c_drift = kappa * dt^alpha, c_stoch = nu * dt^{alpha-1/2}.
        self.c_drift = nn.Parameter(
            torch.tensor(kappa_0 * dt ** alpha, dtype=torch.float32))
        self.c_stoch = nn.Parameter(
            torch.tensor(nu_0 * dt ** (alpha - 0.5), dtype=torch.float32))
        self.theta   = nn.Parameter(torch.tensor(theta_0, dtype=torch.float32))

        # Modal coefficients and rates
        self.a       = nn.Parameter(torch.from_numpy(a_np).float())     # (M,)
        self.log_lam = nn.Parameter(torch.from_numpy(np.log(lam_np)).float())  # (M,)

        self.register_buffer('V_0', torch.tensor(V_0, dtype=torch.float32))
        self.N     = N
        self.M     = M
        self.V_min = V_min

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N) -> V: (B, N+1).  O(M*N*B) total."""
        B = z.shape[0]
        r = torch.exp(-torch.exp(self.log_lam))                    # (M,)  decay factors

        V_list  = [self.V_0.expand(B)]
        S_drift = torch.zeros(B, self.M, device=z.device, dtype=z.dtype)
        S_stoch = torch.zeros(B, self.M, device=z.device, dtype=z.dtype)

        for k in range(self.N):
            V_k    = V_list[k]
            sqV_k  = V_k.clamp(min=0.0).sqrt()
            # State update: each (B, M) tensor
            S_drift = (self.theta - V_k).unsqueeze(1) + r * S_drift
            S_stoch = (sqV_k * z[:, k]).unsqueeze(1) + r * S_stoch
            # Sum across modes weighted by a_m
            drift_sum = (self.a * S_drift).sum(dim=1)              # (B,)
            stoch_sum = (self.a * S_stoch).sum(dim=1)
            V_next    = (self.V_0
                         + self.c_drift * drift_sum
                         + self.c_stoch * stoch_sum).clamp(min=self.V_min)
            V_list.append(V_next)
        return torch.stack(V_list, dim=1)
