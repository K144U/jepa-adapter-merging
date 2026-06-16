"""Minimal, transparent LoRA for arbitrary nn.Linear targets.

Custom rather than peft so the same injector works across timm and HF
encoders and so task vectors (DeltaW = scaling * B @ A) are first-class
objects for the merging and Arm C geometry code.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device  # base may already live on GPU
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features,
                                               device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r,
                                               device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * lora

    def delta_w(self) -> torch.Tensor:
        """DeltaW = scaling * B @ A, shape (out, in), fp32, detached."""
        return (self.scaling * self.lora_B.to(torch.float32)
                @ self.lora_A.to(torch.float32)).detach()


def inject_lora(model: nn.Module, target_names: list, r: int = 16,
                alpha: int = 32, dropout: float = 0.1) -> dict:
    """Replace named nn.Linear modules with LoRALinear. Freezes everything
    else. Returns {name: LoRALinear}."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = {}
    for name in target_names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf = parts[-1]
        base = getattr(parent, leaf)
        assert isinstance(base, nn.Linear), f"{name} is {type(base)}"
        ll = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, leaf, ll)
        wrapped[name] = ll
    return wrapped


def lora_state(wrapped: dict) -> dict:
    """Serializable adapter state: {layer: {A, B, scaling}}."""
    return {name: {"A": m.lora_A.detach().cpu().to(torch.float32),
                   "B": m.lora_B.detach().cpu().to(torch.float32),
                   "scaling": m.scaling}
            for name, m in wrapped.items()}


def save_adapter(wrapped: dict, path, extra: dict = None) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state(wrapped), "extra": extra or {}}, p)


def load_adapter(path) -> dict:
    """Returns the saved {layer: {A, B, scaling}} dict."""
    return torch.load(path, map_location="cpu", weights_only=False)["lora"]


def task_vector(adapter: dict) -> dict:
    """{layer: DeltaW} from a saved adapter state."""
    return {name: s["scaling"] * s["B"] @ s["A"] for name, s in adapter.items()}


def apply_task_vector(model: nn.Module, tv: dict, coef: float = 1.0) -> None:
    """Add coef * DeltaW into the base weights of the named linears, in place.

    Used to materialize a merged encoder for evaluation. Works on a plain
    (non-LoRA-wrapped) model whose module names match the task-vector keys.
    """
    named = dict(model.named_modules())
    for name, dw in tv.items():
        mod = named[name]
        base = mod.base if isinstance(mod, LoRALinear) else mod
        base.weight.data.add_(coef * dw.to(base.weight.dtype).to(base.weight.device))
