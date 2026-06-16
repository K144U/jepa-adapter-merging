"""The five mergers. All operate on task vectors
({layer: DeltaW (out, in)}) and return a merged {layer: DeltaW}.

Deterministic given inputs + config (DARE's mask and rd_encoder's rotation
are seeded). Hyperparameters are tuned once on a held-out validation pair,
then frozen (pre-registration rule).

rd_encoder implements the achievability construction of the companion
rate-distortion view of model merging (Pathak & Garg): the Theorem-4
construction in projector-surrogate geometry with a ridge-regularized
centroid. Here the merged DeltaW is applied directly to the base weights at
eval time, so realization is full-rank (no rank-r truncation step); bits>=32
by default (no quantization).
"""

import hashlib
import math

import torch


def _fp32(tvs):
    return [{k: v.to(torch.float32) for k, v in tv.items()} for tv in tvs]


def merge_uniform(tvs: list, **_) -> dict:
    tvs = _fp32(tvs)
    return {k: sum(tv[k] for tv in tvs) / len(tvs) for k in tvs[0]}


def merge_task_arithmetic(tvs: list, coef: float = 0.3, **_) -> dict:
    tvs = _fp32(tvs)
    return {k: coef * sum(tv[k] for tv in tvs) for k in tvs[0]}


def _ties_combine(stack: torch.Tensor, density: float) -> torch.Tensor:
    """TIES on a (T, ...) stack: trim -> elect sign -> disjoint mean."""
    T = stack.shape[0]
    flat = stack.reshape(T, -1)
    # trim: keep top-density fraction by magnitude per task
    k = max(1, int(density * flat.shape[1]))
    mag = flat.abs()
    thresh = mag.kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values
    trimmed = torch.where(mag >= thresh, flat, torch.zeros_like(flat))
    # elect: sign of the summed trimmed values
    elected = torch.sign(trimmed.sum(dim=0))
    elected[elected == 0] = 1.0
    # disjoint mean over entries agreeing with the elected sign
    agree = (torch.sign(trimmed) == elected.unsqueeze(0)) & (trimmed != 0)
    num = (trimmed * agree).sum(dim=0)
    cnt = agree.sum(dim=0).clamp_min(1)
    return (num / cnt).reshape(stack.shape[1:])


def merge_ties(tvs: list, density: float = 0.2, coef: float = 1.0, **_) -> dict:
    tvs = _fp32(tvs)
    return {k: coef * _ties_combine(torch.stack([tv[k] for tv in tvs]), density)
            for k in tvs[0]}


def merge_dare_ties(tvs: list, drop_p: float = 0.9, density: float = 0.2,
                    coef: float = 1.0, seed: int = 0, **_) -> dict:
    """DARE (random drop + 1/(1-p) rescale per task vector), then TIES."""
    tvs = _fp32(tvs)
    dropped = []
    for t, tv in enumerate(tvs):
        new = {}
        for k, v in tv.items():
            g = torch.Generator().manual_seed(
                int.from_bytes(hashlib.sha256(f"{seed}:{t}:{k}".encode())
                               .digest()[:4], "little"))
            keep = (torch.rand(v.shape, generator=g) >= drop_p).to(v.dtype)
            new[k] = v * keep / (1.0 - drop_p)
        dropped.append(new)
    return merge_ties(dropped, density=density, coef=coef)


# ----- rd_encoder (rate-distortion-optimal merging encoder) ----------------

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _fwht(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    h = 1
    x = x.clone()
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.view(-1, n)
        h *= 2
    return x.view(-1)


def _layer_seed(seed: int, layer: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{layer}".encode())
                          .digest()[:4], "little")


def _right_basis(delta: torch.Tensor, r: int) -> torch.Tensor:
    d = delta.to(torch.float32)
    if min(d.shape) > 256 and r < min(d.shape) // 4:
        _, _, v = torch.svd_lowrank(d, q=min(r + 8, min(d.shape)), niter=4)
        return v[:, :r]
    _, _, vh = torch.linalg.svd(d, full_matrices=False)
    return vh[:r, :].T


def _quantize_rotated(eta, bits, c, seed):
    out_dim, k = eta.shape
    flat = eta.reshape(-1).to(torch.float32)
    n = _next_pow2(flat.numel())
    x = torch.zeros(n, dtype=torch.float32, device=flat.device)
    x[: flat.numel()] = flat
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int8)
             .to(flat.device).to(torch.float32) * 2 - 1)
    xr = _fwht(signs * x) / math.sqrt(n)
    sigma = xr.norm() / math.sqrt(n)
    if sigma < 1e-20:
        return eta
    levels = 2 ** int(round(bits))
    hi = c * sigma
    step = (2 * hi) / levels
    idx = torch.clamp(torch.floor((xr + hi) / step), 0, levels - 1)
    xq = -hi + (idx + 0.5) * step
    back = signs * _fwht(xq) / math.sqrt(n)
    return back[: flat.numel()].reshape(out_dim, k).to(eta.dtype)


def merge_rd_encoder(tvs: list, rank: int = 16, weights: list = None,
                     bits: float = 32, c: float = 5.0, seed: int = 20260611,
                     eig_rel_floor: float = 1e-6, ridge_lambda: float = 0.01,
                     **_) -> dict:
    """Per layer: V_t = top-r right basis of Delta_t; Hbar = sum w_t V_t V_t^T
    eigendecomposed via the Gram trick; ridge-regularized H-weighted centroid
    W* = (sum w_t Delta_t) Q (Lambda + ridge)^{-1} Q^T, optionally Hadamard-
    rotated + quantized in the whitened coordinates when bits < 32.

    ridge_lambda regularizes the centroid: the raw theory centroid (lambda=0)
    can blow up by 25-94x on real adapters (degenerate Hbar spectrum); a small
    ridge interpolates toward task arithmetic. The default here is frozen from
    an independent ridge sweep on held-out data.
    """
    tvs = _fp32(tvs)
    T = len(tvs)
    w = ([1.0 / T] * T if weights is None
         else [float(x) / sum(weights) for x in weights])
    merged = {}
    for layer in tvs[0]:
        deltas = [tv[layer] for tv in tvs]
        r = min(rank, min(deltas[0].shape))
        bases = [_right_basis(d, r) for d in deltas]
        M_w = torch.cat([math.sqrt(wt) * V for wt, V in zip(w, bases)], dim=1)
        G = M_w.T @ M_w
        S, U = torch.linalg.eigh(G)
        keep = S > eig_rel_floor * S.max()
        S, U = S[keep], U[:, keep]
        Q = M_w @ U @ torch.diag(S.rsqrt())
        S_eff = S + float(ridge_lambda)
        N = deltas[0] * w[0]
        for wt, d in zip(w[1:], deltas[1:]):
            N = N + wt * d
        eta = (N @ Q) @ torch.diag(S_eff.rsqrt())
        if bits < 32:
            eta = _quantize_rotated(eta, bits, c, _layer_seed(seed, layer))
        merged[layer] = (eta @ torch.diag(S_eff.rsqrt())) @ Q.T
    return merged


MERGERS = {
    "uniform": merge_uniform,
    "task_arithmetic": merge_task_arithmetic,
    "ties": merge_ties,
    "dare_ties": merge_dare_ties,
    "rd_encoder": merge_rd_encoder,
}
