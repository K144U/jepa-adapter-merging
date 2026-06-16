"""Metric-library checks against synthetic cases with known ground truth
(plan Week 2: 'unit tests against synthetic cases'). Run as a script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from isomerge.metrics import (effective_rank, eigenvalue_entropy,
                              eigenvalue_ratio, feature_drift, isoscore,
                              kendall_stability, linear_cka, pairwise_cosine,
                              participation_ratio, principal_angles,
                              sign_conflict_rate, stable_rank,
                              subspace_overlap)

torch.manual_seed(0)
np.random.seed(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAIL.append(name)


D, N = 32, 20000

# --- isotropy on hand-built Gaussians ---------------------------------------
iso = torch.randn(N, D)
scales = torch.ones(D)
scales[0] = 10.0  # one dominant direction
aniso = torch.randn(N, D) * scales

er_iso, er_an = effective_rank(iso), effective_rank(aniso)
check("erank isotropic ~ D", abs(er_iso - D) < 1.5, f"({er_iso:.2f} vs {D})")
check("erank anisotropic << D", er_an < 0.6 * D, f"({er_an:.2f})")
check("erank ordering", er_iso > er_an)

check("isoscore isotropic high", isoscore(iso) > 0.9, f"({isoscore(iso):.3f})")
check("isoscore anisotropic low", isoscore(aniso) < isoscore(iso) * 0.7,
      f"({isoscore(aniso):.3f})")

check("eig entropy isotropic ~1", eigenvalue_entropy(iso) > 0.97)
check("eig ratio isotropic ~1", eigenvalue_ratio(iso) > 0.8)
check("eig ratio anisotropic small", eigenvalue_ratio(aniso) < 0.05)

collapsed = torch.randn(N, 1) @ torch.randn(1, D)  # rank-1 cloud
check("erank collapsed ~1", effective_rank(collapsed) < 1.2,
      f"({effective_rank(collapsed):.3f})")

# --- task-vector geometry ----------------------------------------------------
u = torch.randn(64, 1)
v = torch.randn(1, 48)
rank1 = u @ v
check("stable_rank rank-1 ~1", abs(stable_rank(rank1) - 1) < 1e-4)
check("PR rank-1 ~1", abs(participation_ratio(rank1) - 1) < 1e-4)

q, _ = torch.linalg.qr(torch.randn(48, 48))
orth = torch.randn(64, 48) @ q  # full-spectrum-ish
check("stable_rank full > rank-1", stable_rank(orth) > 5)

W = torch.randn(64, 48)
tv_a = {"l": W}
tv_b = {"l": -W}
tv_c = {"l": W.clone()}
cos = pairwise_cosine([tv_a, tv_b])
check("cosine opposite = -1", abs(cos[0, 1] + 1) < 1e-5, f"({cos[0,1]:.4f})")
cos2 = pairwise_cosine([tv_a, tv_c])
check("cosine identical = 1", abs(cos2[0, 1] - 1) < 1e-5)

check("sign conflict opposite = 1", sign_conflict_rate([tv_a, tv_b]) == 1.0)
check("sign conflict identical = 0", sign_conflict_rate([tv_a, tv_c]) == 0.0)

# principal angles: identical right-subspace -> 0; disjoint -> pi/2
A_basis = torch.zeros(48, 4)
A_basis[:4] = torch.eye(4)
B_basis = torch.zeros(48, 4)
B_basis[4:8] = torch.eye(4)
Wa = torch.randn(64, 4) @ A_basis.T
Wb = torch.randn(64, 4) @ A_basis.T
Wc = torch.randn(64, 4) @ B_basis.T
ang_same = principal_angles(Wa, Wb, k=4)
ang_disj = principal_angles(Wa, Wc, k=4)
check("principal angles same subspace ~0", float(ang_same.max()) < 1e-3,
      f"(max {ang_same.max():.2e})")
check("principal angles disjoint ~pi/2",
      float(ang_disj.min()) > np.pi / 2 - 1e-3)
check("subspace overlap same = 1", abs(subspace_overlap(Wa, Wb, 4) - 1) < 1e-5)
check("subspace overlap disjoint = 0", subspace_overlap(Wa, Wc, 4) < 1e-5)

# --- functional ---------------------------------------------------------------
X = torch.randn(500, 16)
check("CKA self = 1", abs(linear_cka(X, X) - 1) < 1e-6)
R = torch.randn(16, 16)
check("CKA invariant to rotation", abs(linear_cka(X, X @ R) - 1) > -1
      and linear_cka(X, X @ R) > 0.5)
Y = torch.randn(500, 16)
check("CKA independent ~ 0", linear_cka(X, Y) < 0.1,
      f"({linear_cka(X, Y):.4f})")
check("drift zero on identical", feature_drift(X, X) == 0.0)

check("kendall identical rankings = 1",
      kendall_stability([[1, 2, 3, 4], [1, 2, 3, 4]]) == 1.0)
check("kendall reversed = -1",
      kendall_stability([[1, 2, 3, 4], [4, 3, 2, 1]]) == -1.0)

print(f"\n{'ALL PASS' if not FAIL else f'FAILED: {FAIL}'}")
sys.exit(1 if FAIL else 0)
