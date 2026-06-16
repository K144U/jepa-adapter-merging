"""Arm E: post-hoc ZCA isotropization (plan 1.4).

Embedding-level variant (the committed one): fit ZCA on the backbone's
pooled-embedding distribution, absorb it as a fixed linear map after the
encoder, then run the identical LoRA + merge pipeline on the wrapped
encoder. Weights-only, no retraining.
"""

import torch
import torch.nn as nn


def fit_zca(X: torch.Tensor, eps: float = 1e-3) -> dict:
    """X: (N, D) features. Returns {'W': (D, D), 'mu': (D,)} with
    W = U (S + eps*mean(S))^{-1/2} U^T (shrinkage-stabilized ZCA)."""
    X = X.to(torch.float64)
    mu = X.mean(dim=0)
    Xc = X - mu
    C = Xc.T @ Xc / max(len(X) - 1, 1)
    S, U = torch.linalg.eigh(C)
    S = S.clamp_min(0.0) + eps * S.mean()
    W = U @ torch.diag(S.rsqrt()) @ U.T
    return {"W": W.to(torch.float32), "mu": mu.to(torch.float32)}


class WhitenedEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, zca: dict):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("W", zca["W"])
        self.register_buffer("mu", zca["mu"])
        self.embed_dim = encoder.embed_dim
        self.img_size = encoder.img_size
        self.norm_mean = encoder.norm_mean
        self.norm_std = encoder.norm_std

    def forward(self, x):
        z = self.encoder(x)
        return (z - self.mu) @ self.W.T

    def lora_targets(self):
        return ["encoder." + n for n in self.encoder.lora_targets()]
