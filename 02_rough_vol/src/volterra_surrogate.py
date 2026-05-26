"""
volterra_surrogate.py  —  learnable Volterra-aware surrogate for rough Heston.

Architecture (autoregressive, V-conditional):

    V_{k+1} = V_0
              + c_drift * sum_{j=0..k} w[k-j] * (theta - V_j)
              + c_stoch * sum_{j=0..k} w[k-j] * sqrt(V_j) * z_j
              clipped to [V_min, +inf)

All of (theta, c_drift, c_stoch, w[i] for i=0..N-1) are learnable.

Initialised at rough Heston exactly:
    w[i]    = (i+1)^{alpha-1}        (RL kernel, alpha = H + 0.5)
    c_drift = kappa * dt^alpha / Gamma(alpha)
    c_stoch = nu    * dt^{alpha-0.5} / Gamma(alpha)
    theta   = theta_0

This is the architecture §5.4 prescribes to close the residual cross-
model bias: a V-conditional Volterra update (sqrt(V) in the diffusion)
that cannot be expressed as a linear filter of z.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma as _gamma


class VolterraSurrogate(nn.Module):
    def __init__(self, N: int, dt: float, H: float = 0.1,
                 V_0: float = 0.04, theta_0: float = 0.04,
                 kappa_0: float = 2.0, nu_0: float = 1.9,
                 V_min: float = 0.0):
        super().__init__()
        alpha = H + 0.5
        # Learnable scalar drift/diffusion rates
        self.c_drift = nn.Parameter(
            torch.tensor(kappa_0 * dt ** alpha / _gamma(alpha), dtype=torch.float32))
        self.c_stoch = nn.Parameter(
            torch.tensor(nu_0 * dt ** (alpha - 0.5) / _gamma(alpha), dtype=torch.float32))
        self.theta   = nn.Parameter(torch.tensor(theta_0, dtype=torch.float32))
        # Learnable Volterra kernel weights w[i] = (i+1)^{alpha-1}, length N
        i_arr  = np.arange(N, dtype=np.float64)
        w_init = (i_arr + 1.0) ** (alpha - 1.0)
        self.w = nn.Parameter(torch.from_numpy(w_init).float())
        # Fixed
        self.register_buffer('V_0', torch.tensor(V_0, dtype=torch.float32))
        self.N     = N
        self.V_min = V_min

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N)  i.i.d. N(0,1)  -- causal BM innovations
        Returns V: (B, N+1) variance paths.

        Built via list+stack (not in-place) so autograd works through
        the Volterra recursion.
        """
        B = z.shape[0]
        V_list = [self.V_0.expand(B)]                          # V[0]
        for k in range(self.N):
            V_curr = torch.stack(V_list, dim=1)                # (B, k+1)
            wk        = self.w[:k + 1].flip(0)                  # (k+1,)
            drift_sum = ((self.theta - V_curr) * wk).sum(dim=1)
            sqV       = V_curr.clamp(min=0.0).sqrt()
            stoch_sum = (sqV * z[:, :k + 1] * wk).sum(dim=1)
            V_next    = (self.V_0
                         + self.c_drift * drift_sum
                         + self.c_stoch * stoch_sum).clamp(min=self.V_min)
            V_list.append(V_next)
        return torch.stack(V_list, dim=1)                       # (B, N+1)
