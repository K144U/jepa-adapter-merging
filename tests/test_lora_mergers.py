"""LoRA injection/extraction roundtrip + merger correctness on hand-built
task vectors."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from isomerge.adapt.lora import (LoRALinear, apply_task_vector, inject_lora,
                                 load_adapter, save_adapter, task_vector)
from isomerge.merging import MERGERS

torch.manual_seed(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAIL.append(name)


# --- LoRA --------------------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


net = Tiny()
x = torch.randn(4, 8)
y0 = net(x)
wrapped = inject_lora(net, ["fc1", "fc2"], r=2, alpha=4, dropout=0.0)
check("LoRA identity at init (B=0)", torch.allclose(net(x), y0, atol=1e-6))
check("base frozen", not net.fc1.base.weight.requires_grad)
check("lora trainable", net.fc1.lora_A.requires_grad)

with torch.no_grad():
    for m in wrapped.values():
        m.lora_B.normal_()
y1 = net(x)
check("LoRA changes output once B != 0", not torch.allclose(y1, y0))

# delta_w equals the functional effect on the linear path
lin_in = torch.randn(4, 8)
m = wrapped["fc1"]
direct = m(lin_in) - m.base(lin_in)
via_dw = lin_in @ m.delta_w().T
check("delta_w matches forward effect",
      torch.allclose(direct, via_dw, atol=1e-5))

# save / load / apply roundtrip
p = ROOT / "results" / "tmp_test_adapter.pt"
save_adapter(wrapped, p, extra={"task": "t"})
tv = task_vector(load_adapter(p))
plain = Tiny()
plain.load_state_dict({k.replace("base.", ""): v
                       for k, v in net.state_dict().items()
                       if "lora" not in k})
y_lora = net(x)
apply_task_vector(plain, tv)
check("apply_task_vector reproduces LoRA model",
      torch.allclose(plain(x), y_lora, atol=1e-5),
      f"(max diff {(plain(x) - y_lora).abs().max():.2e})")
p.unlink()

# --- mergers -----------------------------------------------------------------
def rand_tv(rank=4, out=32, inn=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(out, rank, generator=g) @ torch.randn(rank, inn, generator=g),
            "b": torch.randn(out, rank, generator=g) @ torch.randn(rank, inn, generator=g)}


t1, t2 = rand_tv(seed=1), rand_tv(seed=2)

u = MERGERS["uniform"]([t1, t2])
check("uniform = mean",
      torch.allclose(u["a"], (t1["a"] + t2["a"]) / 2, atol=1e-5))

ta = MERGERS["task_arithmetic"]([t1, t2], coef=0.5)
check("TA = coef * sum",
      torch.allclose(ta["a"], 0.5 * (t1["a"] + t2["a"]), atol=1e-5))

# TIES: identical task vectors at density 1 -> the vector itself
same = MERGERS["ties"]([t1, {k: v.clone() for k, v in t1.items()}],
                       density=1.0, coef=1.0)
check("TIES identical inputs -> identity",
      torch.allclose(same["a"], t1["a"], atol=1e-5))

# TIES: exactly opposed vectors never average to zero against the elected sign
opp = MERGERS["ties"]([t1, {k: -v for k, v in t1.items()}], density=1.0)
check("TIES opposed: elected magnitudes kept",
      torch.allclose(opp["a"].abs(), t1["a"].abs(), atol=1e-5))

# DARE preserves expected scale (loose statistical check)
d = MERGERS["dare_ties"]([t1, t2], drop_p=0.5, density=1.0, seed=3)
ratio = float(d["a"].norm() / u["a"].norm())
check("DARE-TIES scale sane", 0.3 < ratio < 4.0, f"(ratio {ratio:.2f})")

# rd_encoder, single task, no ridge: projector centroid reconstructs delta
rd1 = MERGERS["rd_encoder"]([t1], rank=4, ridge_lambda=0.0)
err = float((rd1["a"] - t1["a"]).norm() / t1["a"].norm())
check("rd_encoder T=1 reconstructs", err < 1e-4, f"(rel err {err:.2e})")

# rd_encoder two tasks: finite, right shape, ridge tames norm
rd2 = MERGERS["rd_encoder"]([t1, t2], rank=4, ridge_lambda=0.01)
rd2_raw = MERGERS["rd_encoder"]([t1, t2], rank=4, ridge_lambda=0.0)
check("rd_encoder finite", all(torch.isfinite(v).all() for v in rd2.values()))
check("ridge reduces centroid norm",
      sum(float(v.norm()) for v in rd2.values())
      <= sum(float(v.norm()) for v in rd2_raw.values()) + 1e-6)

# quantized path runs and stays close at high bits
rdq = MERGERS["rd_encoder"]([t1, t2], rank=4, ridge_lambda=0.01, bits=8)
rel = (sum(float((rdq[k] - rd2[k]).norm()) for k in rd2)
       / sum(float(rd2[k].norm()) for k in rd2))
check("rd_encoder 8-bit close to fp", rel < 0.1, f"(rel {rel:.3f})")

# determinism
d2 = MERGERS["dare_ties"]([t1, t2], drop_p=0.5, density=1.0, seed=3)
check("DARE deterministic given seed",
      all(torch.equal(d[k], d2[k]) for k in d))

print(f"\n{'ALL PASS' if not FAIL else f'FAILED: {FAIL}'}")
sys.exit(1 if FAIL else 0)
