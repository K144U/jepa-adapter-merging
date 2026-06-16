"""SIGReg sanity: discriminates Gaussian from collapsed/scaled/shifted
distributions, is differentiable, and agrees in ordering with the vendored
LeJEPA reference implementation when available."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from isomerge.adapt.sigreg import SIGReg

torch.manual_seed(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAIL.append(name)


sig = SIGReg(num_slices=512, resample=False, seed=1)
N, D = 4096, 64

gauss = torch.randn(N, D)
collapsed = torch.randn(N, 1).expand(N, D).contiguous()
scaled = 5.0 * torch.randn(N, D)
shifted = torch.randn(N, D) + 3.0
aniso = torch.randn(N, D) * torch.linspace(0.1, 5, D)

s_g = float(sig(gauss))
s_c = float(sig(collapsed))
s_s = float(sig(scaled))
s_m = float(sig(shifted))
s_a = float(sig(aniso))
print(f"  stats: gauss={s_g:.5f} collapsed={s_c:.5f} scaled={s_s:.5f} "
      f"shifted={s_m:.5f} aniso={s_a:.5f}")

check("gauss lowest", s_g < min(s_c, s_s, s_m, s_a))
check("collapsed >> gauss", s_c > 10 * s_g)
check("anisotropic > gauss", s_a > 3 * s_g)
check("nonnegative", min(s_g, s_c, s_s, s_m, s_a) >= 0)

z = torch.randn(256, 32, requires_grad=True)
loss = sig(z)
loss.backward()
check("differentiable", z.grad is not None and torch.isfinite(z.grad).all())

# minimizing SIGReg from an anisotropic start should raise isotropy
sys.path.insert(0, str(ROOT / "src"))
from isomerge.metrics import isoscore  # noqa: E402

x = (torch.randn(2048, 16) * torch.linspace(0.05, 3, 16)).requires_grad_(True)
opt = torch.optim.Adam([x], lr=0.05)
iso_before = isoscore(x.detach())
sgd_sig = SIGReg(num_slices=128, resample=True, seed=2)
for _ in range(150):
    opt.zero_grad()
    sgd_sig(x).backward()
    opt.step()
iso_after = isoscore(x.detach())
check("minimizing SIGReg raises isotropy", iso_after > iso_before + 0.2,
      f"({iso_before:.3f} -> {iso_after:.3f})")

# ordering agreement with the vendored LeJEPA reference, if importable
try:
    sys.path.insert(0, str(ROOT / "third_party" / "lejepa"))
    from lejepa.univariate.epps_pulley import EppsPulley  # noqa: E402
    ref = EppsPulley()
    r_g = float(ref(gauss[:, 0:1]).mean())
    r_c = float(ref(collapsed[:, 0:1] * 0 + 1.0).mean())
    check("reference EP ordering matches", (r_c > r_g) == (s_c > s_g),
          f"(ref gauss={r_g:.4f} const={r_c:.4f})")
except Exception as e:  # noqa: BLE001
    print(f"  SKIP  vendored lejepa cross-check ({type(e).__name__}: {e})")

print(f"\n{'ALL PASS' if not FAIL else f'FAILED: {FAIL}'}")
sys.exit(1 if FAIL else 0)
