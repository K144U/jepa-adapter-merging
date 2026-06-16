"""SIGReg: sketched isotropic-Gaussian regularization (LeJEPA, Balestriero &
LeCun 2025).

Pushes a batch of embeddings toward N(0, I) by projecting onto random unit
directions and penalizing each 1D projection's Epps-Pulley statistic against
the standard normal:

    T(u) = integral over t of |phi_emp(t; u^T z) - exp(-t^2/2)|^2 w(t) dt
    SIGReg = mean over num_slices random directions u of T(u)

w(t) is the N(0,1) density; the integral uses Gauss-Hermite quadrature.
Self-contained reimplementation (~50 lines, as the plan budgeted); the
vendored reference is third_party/lejepa (tests cross-check against it when
importable). Fully differentiable; linear in batch size and dimension.
"""

import math

import numpy as np
import torch


class SIGReg(torch.nn.Module):
    def __init__(self, num_slices: int = 256, n_points: int = 17,
                 resample: bool = True, seed: int = 0):
        super().__init__()
        self.num_slices = num_slices
        self.resample = resample
        self._step = 0
        self._seed = seed
        # Gauss-Hermite: integral g(t) exp(-t^2) dt ~= sum w_i g(t_i).
        # Target integral has weight N(0,1) = exp(-t^2/2)/sqrt(2pi); substitute
        # t = sqrt(2) s so weights fold to w_i / sqrt(pi) at nodes sqrt(2) s_i.
        nodes, weights = np.polynomial.hermite.hermgauss(n_points)
        t = torch.tensor(nodes * math.sqrt(2.0), dtype=torch.float32)
        w = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float32)
        self.register_buffer("t", t)
        self.register_buffer("w", w)
        self.register_buffer("phi_target", torch.exp(-t ** 2 / 2))

    def directions(self, dim: int, device) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self._seed + (self._step if self.resample else 0))
        A = torch.randn(dim, self.num_slices, generator=g)
        A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return A.to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, D) embeddings. Returns scalar SIGReg loss."""
        A = self.directions(z.shape[-1], z.device).to(z.dtype)
        self._step += 1
        x = z @ A                                   # (N, K)
        tx = x.unsqueeze(-1) * self.t.to(z.dtype)   # (N, K, P)
        phi_re = torch.cos(tx).mean(dim=0)          # (K, P) empirical char. fn
        phi_im = torch.sin(tx).mean(dim=0)
        err = (phi_re - self.phi_target.to(z.dtype)) ** 2 + phi_im ** 2
        stat = (err * self.w.to(z.dtype)).sum(dim=-1)   # (K,)
        return stat.mean()
