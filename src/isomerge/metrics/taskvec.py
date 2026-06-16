"""Task-vector geometry metrics (plan section 1.3, 'Task-vector geometry').

A task vector is a dict {layer_name: DeltaW tensor (out, in)} -- the weight
update each LoRA adapter induces (DeltaW = scaling * B @ A).
"""

import numpy as np
import torch


def flatten_taskvec(tv: dict) -> torch.Tensor:
    """Concatenate all layer deltas into one flat fp32 vector (sorted by name)."""
    return torch.cat([tv[k].reshape(-1).to(torch.float32) for k in sorted(tv)])


def pairwise_cosine(tvs: list) -> np.ndarray:
    """(T, T) cosine-similarity matrix of flattened task vectors."""
    flats = [flatten_taskvec(tv) for tv in tvs]
    M = torch.stack([f / f.norm().clamp_min(1e-12) for f in flats])
    return (M @ M.T).numpy()


def sign_conflict_rate(tvs: list, tau: float = 0.0) -> float:
    """Fraction of parameters where task vectors disagree in sign.

    A parameter conflicts if at least two task vectors have opposite nonzero
    signs there (|value| > tau counts as nonzero).
    Conflict = both positive and negative entries present.
    """
    flats = torch.stack([flatten_taskvec(tv) for tv in tvs])  # (T, P)
    pos = (flats > tau).any(dim=0)
    neg = (flats < -tau).any(dim=0)
    active = ((flats.abs() > tau).any(dim=0)).float().sum().clamp_min(1.0)
    return float((pos & neg).float().sum() / active)


def per_layer_sign_conflict(tvs: list, tau: float = 0.0) -> dict:
    out = {}
    for k in sorted(tvs[0]):
        out[k] = sign_conflict_rate([{k: tv[k]} for tv in tvs], tau)
    return out


def stable_rank(W: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2."""
    W = W.to(torch.float32)
    fro2 = float((W ** 2).sum())
    if fro2 <= 0:
        return 0.0
    top = float(torch.linalg.matrix_norm(W, ord=2))
    return fro2 / max(top ** 2, 1e-30)


def participation_ratio(W: torch.Tensor) -> float:
    """PR of the squared singular values: (sum s^2)^2 / sum s^4."""
    s = torch.linalg.svdvals(W.to(torch.float32))
    s2 = s ** 2
    denom = float((s2 ** 2).sum())
    if denom <= 0:
        return 0.0
    return float(s2.sum() ** 2 / denom)


def principal_angles(Wi: torch.Tensor, Wj: torch.Tensor, k: int = 8) -> np.ndarray:
    """Principal angles (radians, ascending) between top-k right singular
    subspaces of two deltas. Do tasks write into the same input directions?"""
    def right_basis(W, k):
        _, _, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
        return Vh[: min(k, Vh.shape[0])].T  # (in, k)
    Vi, Vj = right_basis(Wi, k), right_basis(Wj, k)
    s = torch.linalg.svdvals(Vi.T @ Vj).clamp(-1.0, 1.0)
    return np.arccos(s.numpy())


def subspace_overlap(Wi: torch.Tensor, Wj: torch.Tensor, k: int = 8) -> float:
    """Mean squared cosine of principal angles in [0, 1] (1 = same subspace)."""
    ang = principal_angles(Wi, Wj, k)
    return float(np.mean(np.cos(ang) ** 2))


def backbone_alignment(W: torch.Tensor, feat_eigvecs: torch.Tensor,
                       k: int = 8) -> float:
    """H3's sharpest prediction: energy fraction of DeltaW's input action that
    falls inside the backbone covariance's top-k eigendirections.

    feat_eigvecs: (D, k) top eigenvectors of the layer-input feature covariance.
    Returns ||W U||_F^2 / ||W||_F^2 in [0, 1]; high on anisotropic backbones
    if task updates crowd into dominant directions.
    """
    W = W.to(torch.float32)
    U = feat_eigvecs[:, :k].to(torch.float32)
    fro2 = float((W ** 2).sum())
    if fro2 <= 0:
        return 0.0
    return float(((W @ U) ** 2).sum() / fro2)


def taskvec_summary(tvs: list, names: list = None, k: int = 8) -> dict:
    """Full Arm C task-vector report for one adapter set."""
    names = names or [f"task{i}" for i in range(len(tvs))]
    layers = sorted(tvs[0])
    cos = pairwise_cosine(tvs)
    T = len(tvs)
    overlaps, angles_deg = [], []
    for i in range(T):
        for j in range(i + 1, T):
            ov = np.mean([subspace_overlap(tvs[i][l], tvs[j][l], k) for l in layers])
            overlaps.append(float(ov))
    sr = {n: float(np.mean([stable_rank(tv[l]) for l in layers]))
          for n, tv in zip(names, tvs)}
    pr = {n: float(np.mean([participation_ratio(tv[l]) for l in layers]))
          for n, tv in zip(names, tvs)}
    iu = np.triu_indices(T, 1)
    return {
        "names": names,
        "pairwise_cosine_mean": float(cos[iu].mean()) if T > 1 else 0.0,
        "pairwise_cosine": cos.tolist(),
        "sign_conflict_rate": sign_conflict_rate(tvs),
        "sign_conflict_per_layer": per_layer_sign_conflict(tvs),
        "subspace_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "stable_rank": sr,
        "participation_ratio": pr,
    }
