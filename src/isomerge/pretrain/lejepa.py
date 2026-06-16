"""LeJEPA-style ViT-S pretraining with the SIGReg lambda dial (Arm B).

Objective (LeJEPA, Balestriero & LeCun 2025): V augmented views per image,
one shared encoder, no predictor / EMA teacher / stop-gradient:

    L = L_pred + lambda * SIGReg
    L_pred  = mean_v || z_v - z_bar ||^2 / D   (views pull to their mean)
    SIGReg  = sliced Epps-Pulley statistic of all view embeddings vs N(0, I)

lambda is THE Arm B dial: {0, 0.1, 0.3, 1.0, 3.0} x LAMBDA_REPO_DEFAULT.
The public repo ships only the statistics package, not the training loop
(checked 2026-06-12), so this loop follows the paper recipe; the statistic
itself is cross-checked against the vendored package in tests.

Resume-safe across 24h PBS walltime kills: checkpoints every epoch and every
SAVE_EVERY_STEPS; rerunning the same command continues from the newest.
"""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..adapt.sigreg import SIGReg
from ..utils import amp_dtype, device as _device, save_json, seed_everything

# Paper-default SIGReg coefficient; the repo's IN-10 ablation sweeps
# {0.01..0.1}. Verify against the GPU pilot in Week 1 before the full sweep.
LAMBDA_REPO_DEFAULT = 0.05
SAVE_EVERY_STEPS = 1000


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, base, view_transform, n_views: int = 2):
        self.base = base
        self.t = view_transform
        self.n_views = n_views

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        if torch.is_tensor(img):           # synthetic toy path: jitter crops
            views = [img + 0.05 * torch.randn_like(img)
                     for _ in range(self.n_views)]
        else:
            views = [self.t(img) for _ in range(self.n_views)]
        return torch.stack(views)


def make_view_transform(img_size: int):
    import torchvision.transforms as T
    # GaussianBlur kernel kept small: a large CPU blur (img_size//10*2+1 ~= 45
    # at 224px) was the throughput bottleneck (~8.9 s/step on ViT-S, CPU-aug-
    # bound). A 7px blur preserves blur-invariance at ~40x lower CPU cost.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=7)], p=0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def pretrain_lejepa(dataset, out_dir, lambda_mult: float = 1.0,
                    model_name: str = "vit_small_patch16_224",
                    img_size: int = 224, epochs: int = 200,
                    batch_size: int = 256, n_views: int = 2,
                    lr: float = 1e-3, weight_decay: float = 0.05,
                    warmup_epochs: int = 10, seed: int = 0,
                    num_workers: int = 8, max_steps: int = None,
                    embed_via_pool: bool = True) -> dict:
    """Train one Arm B backbone. lambda_mult is in units of the repo default
    (plan table: 0, 0.1, 0.3, 1.0, 3.0). Writes checkpoints + final encoder
    to out_dir; resumes automatically if a checkpoint exists."""
    import timm
    seed_everything(seed)
    dev = _device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lam = lambda_mult * LAMBDA_REPO_DEFAULT

    net = timm.create_model(model_name, pretrained=False, num_classes=0,
                            img_size=img_size).to(dev)
    n_prefix = getattr(net, "num_prefix_tokens", 1)

    def embed(x):
        f = net.forward_features(x)
        return f[:, n_prefix:].mean(dim=1) if embed_via_pool else f[:, 0]

    sigreg = SIGReg(num_slices=256, seed=seed).to(dev)
    # Anti-collapse normalization (LeJEPA README: BatchNorm on the pooled
    # tokens before the loss). Without it the objective has a trivial
    # collapse minimum -- embeddings shrink toward a point, SIGReg goes flat
    # at its zero-embedding value (~0.163) with no gradient to re-expand, and
    # the lambda dial never engages. BatchNorm(affine=False) forces unit
    # variance per dim across the batch, so collapse-to-a-point is impossible
    # and SIGReg meaningfully measures isotropy of the normalized features.
    embed_bn = torch.nn.BatchNorm1d(net.num_features, affine=False).to(dev)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        pin_memory=dev.type == "cuda",
                        persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)
    opt = torch.optim.AdamW(net.parameters(), lr=lr,
                            weight_decay=weight_decay)
    steps_per_epoch = max(len(loader), 1)
    total_steps = max_steps or epochs * steps_per_epoch
    warmup = max(1, warmup_epochs * steps_per_epoch)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    ckpt_path = out_dir / "checkpoint.pt"
    start_step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["encoder"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        sigreg._step = ck.get("sigreg_step", start_step)
        print(f"[pretrain] resumed at step {start_step}/{total_steps}", flush=True)

    def save(step, final=False):
        torch.save({"encoder": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "sigreg_step": sigreg._step,
                    "lambda_mult": lambda_mult, "lambda": lam},
                   ckpt_path)
        if final:
            torch.save({"encoder": net.state_dict(),
                        "lambda_mult": lambda_mult, "lambda": lam,
                        "model_name": model_name, "img_size": img_size,
                        "steps": step},
                       out_dir / "encoder_final.pt")

    dtype = amp_dtype()
    step = start_step
    t0 = time.time()
    log = []
    net.train()
    while step < total_steps:
        for views in loader:
            if step >= total_steps:
                break
            B, V = views.shape[0], views.shape[1]
            x = views.to(dev).flatten(0, 1)            # (B*V, 3, H, W)
            with torch.autocast(dev.type, dtype=dtype,
                                enabled=dtype != torch.float32):
                z_flat = embed_bn(embed(x).float())    # (B*V, D), unit-var/dim
                z = z_flat.view(B, V, -1)              # (B, V, D)
                z_bar = z.mean(dim=1, keepdim=True)
                l_pred = ((z - z_bar) ** 2).mean()
                l_sig = sigreg(z_flat)
                loss = l_pred + lam * l_sig
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == total_steps:
                rec = {"step": step, "loss": float(loss), "pred": float(l_pred),
                       "sigreg": float(l_sig),
                       "min": (time.time() - t0) / 60}
                log.append(rec)
                print(f"[pretrain lam{lambda_mult}x] {rec}", flush=True)
            if step % SAVE_EVERY_STEPS == 0:
                save(step)
    save(step, final=True)
    save_json({"lambda_mult": lambda_mult, "lambda": lam, "steps": step,
               "log_tail": log[-20:]}, out_dir / "pretrain_summary.json")
    return {"encoder": net, "steps": step, "out_dir": str(out_dir)}


def get_pretrain_dataset(name: str, img_size: int = 224, n_views: int = 2):
    """imagenet100 (HF clane9/imagenet-100), cifar100 (smoke), toy (CPU)."""
    t = make_view_transform(img_size)
    if name == "toy":
        from ..data.tasks import SyntheticTask
        base = SyntheticTask(99, "train", n_classes=4, n_per_class=64,
                             img_size=img_size)
        return MultiViewDataset(base, t, n_views)
    if name == "cifar100":
        import torchvision.datasets as tvd
        from ..data.tasks import DATA_ROOT
        base = tvd.CIFAR100(str(DATA_ROOT), train=True, download=True)
        return MultiViewDataset(base, t, n_views)
    if name == "imagenet100":
        from ..data.tasks import HFImageDataset
        base = HFImageDataset("clane9/imagenet-100", "train")
        return MultiViewDataset(base, t, n_views)
    raise KeyError(name)
