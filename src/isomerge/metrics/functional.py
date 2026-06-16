"""Functional-interference metrics (plan section 1.3, 'Functional interference')."""

import numpy as np
import torch
from scipy.stats import kendalltau


def feature_drift(f_merged: torch.Tensor, f_solo: torch.Tensor) -> float:
    """Cross-task feature drift: mean ||f_merged(x) - f_i(x)|| / ||f_i(x)||
    over task-i data. Inputs: (N, D) features from the same inputs."""
    num = (f_merged.to(torch.float32) - f_solo.to(torch.float32)).norm(dim=1)
    den = f_solo.to(torch.float32).norm(dim=1).clamp_min(1e-12)
    return float((num / den).mean())


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two representations of the same inputs, (N, D)/(N, D')."""
    X = X.to(torch.float64) - X.to(torch.float64).mean(0, keepdim=True)
    Y = Y.to(torch.float64) - Y.to(torch.float64).mean(0, keepdim=True)
    xy = float((X.T @ Y).norm() ** 2)
    xx = float((X.T @ X).norm())
    yy = float((Y.T @ Y).norm())
    if xx <= 0 or yy <= 0:
        return 0.0
    return xy / (xx * yy)


def kendall_stability(rankings: list) -> float:
    """Mean pairwise Kendall tau across accuracy rankings (lists of equal
    length, e.g. per-task accuracy vectors from different seeds/mergers)."""
    taus = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            t, _ = kendalltau(rankings[i], rankings[j])
            if not np.isnan(t):
                taus.append(t)
    return float(np.mean(taus)) if taus else 1.0
