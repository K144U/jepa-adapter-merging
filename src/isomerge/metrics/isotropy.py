"""Backbone-geometry isotropy metrics (plan section 1.3, 'Backbone geometry').

All functions take a feature matrix X of shape (N, D) (numpy or torch) and
return floats. Eigen-quantities are computed on the covariance of X.
"""

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def cov_eigenvalues(X) -> np.ndarray:
    """Eigenvalues of the feature covariance, descending, clipped at 0."""
    X = _to_numpy(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    n = max(X.shape[0] - 1, 1)
    if X.shape[0] >= X.shape[1]:
        C = Xc.T @ Xc / n
        ev = np.linalg.eigvalsh(C)
    else:
        # Gram trick: nonzero eigenvalues of cov match those of Xc Xc^T / n
        G = Xc @ Xc.T / n
        ev = np.linalg.eigvalsh(G)
    ev = np.clip(ev, 0.0, None)[::-1]
    return ev


def effective_rank(X=None, eigenvalues=None) -> float:
    """erank = exp(H(p)), p = normalized covariance eigenvalues (Roy & Vetterli)."""
    ev = cov_eigenvalues(X) if eigenvalues is None else np.asarray(eigenvalues, float)
    s = ev.sum()
    if s <= 0:
        return 1.0
    p = ev / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def eigenvalue_entropy(X=None, eigenvalues=None) -> float:
    """Normalized spectral entropy in [0, 1]: H(p) / log(D)."""
    ev = cov_eigenvalues(X) if eigenvalues is None else np.asarray(eigenvalues, float)
    s = ev.sum()
    if s <= 0 or len(ev) < 2:
        return 0.0
    p = ev / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(ev)))


def eigenvalue_ratio(X=None, eigenvalues=None, floor: float = 1e-12) -> float:
    """min/max covariance eigenvalue ratio (1 = perfectly isotropic)."""
    ev = cov_eigenvalues(X) if eigenvalues is None else np.asarray(eigenvalues, float)
    if len(ev) == 0 or ev[0] <= 0:
        return 0.0
    return float(max(ev[-1], 0.0) / max(ev[0], floor))


def isoscore(X) -> float:
    """IsoScore (Rudman et al., 2022): isotropy of the point cloud in [~0, 1].

    Steps: PCA-reorient, take the diagonal of the covariance in the PCA basis,
    normalize to length sqrt(D), measure distance to the isotropy baseline,
    rescale to fraction-of-dimensions-used, then linearly map to [0, 1].
    """
    X = _to_numpy(X)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    # diagonal of covariance in the PCA basis == eigenvalues (variances)
    ev = cov_eigenvalues(Xc)
    var = np.zeros(d)
    var[: len(ev)] = ev
    norm = np.linalg.norm(var)
    if norm <= 0:
        return 0.0
    var_hat = np.sqrt(d) * var / norm
    # isotropy defect: distance to the all-ones vector
    delta = np.linalg.norm(var_hat - np.ones(d)) / np.sqrt(2 * (d - np.sqrt(d)))
    delta = min(delta, 1.0)
    phi = (1 - delta) ** 2  # fraction of dimensions isotropically used
    score = (d * phi - 1) / (d - 1)
    return float(max(score, 0.0))


def isotropy_profile(X) -> dict:
    """All isotropy metrics from one eigendecomposition."""
    ev = cov_eigenvalues(X)
    d = _to_numpy(X).shape[1]
    return {
        "effective_rank": effective_rank(eigenvalues=ev),
        "effective_rank_frac": effective_rank(eigenvalues=ev) / d,
        "eigenvalue_entropy": eigenvalue_entropy(eigenvalues=ev),
        "eigenvalue_ratio": eigenvalue_ratio(eigenvalues=ev),
        "isoscore": isoscore(X),
        "dim": d,
        "top1_eig_frac": float(ev[0] / ev.sum()) if ev.sum() > 0 else 1.0,
    }
