"""LoRA fine-tuning with optional SIGReg (conditions C0/C1, plan 1.1).

One call = one (backbone, task, condition, seed) cell. Heads are task-private
and never merged. Cells are atomic for the orchestrator: ~1h each, retried on
failure, so there is no mid-run checkpointing here (pretraining has it).
"""

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.tasks import get_task, make_transform, n_classes
from ..utils import amp_dtype, device as _device, seed_everything
from .lora import inject_lora, save_adapter
from .sigreg import SIGReg


def finetune(encoder, task: str, lambda_ft: float = 0.0, seed: int = 0,
             r: int = 16, alpha: int = 32, dropout: float = 0.1,
             epochs: int = 10, batch_size: int = 128, lr: float = 1e-4,
             weight_decay: float = 0.01, warmup_frac: float = 0.05,
             max_steps: int = None, num_workers: int = 4,
             out_path=None) -> dict:
    """Fine-tune LoRA on `task`. lambda_ft > 0 = C1 (SIGReg-LoRA), 0 = C0.

    Returns {"adapter", "head", "test_acc", ...}; saves to out_path if given.
    """
    seed_everything(seed)
    dev = _device()
    encoder = encoder.to(dev)
    wrapped = inject_lora(encoder, encoder.lora_targets(), r=r, alpha=alpha,
                          dropout=dropout)
    head = nn.Linear(encoder.embed_dim, n_classes(task)).to(dev)
    sigreg = SIGReg(seed=seed).to(dev) if lambda_ft > 0 else None

    train_ds = get_task(task, "train", make_transform(encoder, train=True))
    test_ds = get_task(task, "test", make_transform(encoder, train=False))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=len(train_ds) > batch_size,
                        pin_memory=dev.type == "cuda")

    params = [p for m in wrapped.values() for p in (m.lora_A, m.lora_B)]
    params += list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # max_steps is an UPPER BOUND on the epoch schedule: small datasets still
    # run their full `epochs`, large ones are capped (min, not override) so a
    # single big task can't dominate a pilot. (Plain override would over-train
    # tiny datasets to max_steps epochs.)
    total_steps = epochs * max(len(loader), 1)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    dtype = amp_dtype()
    ce = nn.CrossEntropyLoss()
    step, t0 = 0, time.time()
    encoder.train()
    head.train()
    done = False
    for _epoch in range(10 ** 9 if max_steps else epochs):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, dtype=dtype,
                                enabled=dtype != torch.float32):
                z = encoder(x)
                loss = ce(head(z), y)
                if sigreg is not None:
                    loss = loss + lambda_ft * sigreg(z.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[ft {task} seed{seed}] step {step}/{total_steps} "
                      f"loss {loss.item():.4f}", flush=True)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    test_acc = evaluate(encoder, head, test_ds, batch_size, num_workers)
    result = {
        "task": task, "lambda_ft": lambda_ft, "seed": seed, "r": r,
        "alpha": alpha, "steps": step, "test_acc": test_acc,
        "minutes": (time.time() - t0) / 60,
    }
    if out_path:
        save_adapter(wrapped, out_path,
                     extra={**result,
                            "head_w": head.weight.detach().cpu(),
                            "head_b": head.bias.detach().cpu()})
    print(f"[ft {task} seed{seed}] test_acc {test_acc:.4f}", flush=True)
    return {**result, "wrapped": wrapped, "head": head}


@torch.no_grad()
def evaluate(encoder, head, ds, batch_size=128, num_workers=4) -> float:
    dev = next(head.parameters()).device
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    encoder.eval()
    head.eval()
    correct = total = 0
    for x, y in loader:
        pred = head(encoder(x.to(dev))).argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    encoder.train()
    head.train()
    return correct / max(total, 1)
