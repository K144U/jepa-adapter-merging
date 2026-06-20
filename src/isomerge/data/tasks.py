"""The 8-task vision-merging suite (plan section 1.1) + synthetic toy tasks.

Sources: torchvision where available (DTD, EuroSAT, GTSRB, MNIST, SUN397,
SVHN); HuggingFace datasets for Stanford Cars (torchvision's download is
dead) and RESISC45 (not in torchvision) via the tanganke/* mirrors used by
the model-merging literature (FusionBench).

get_task(name, split, transform) -> torch.utils.data.Dataset with .classes.
Splits: 'train', 'val' (10% carved from train, seed 0), 'test'.
Synthetic tasks 'toy0'..'toyN' need no download (CPU pipeline tests).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

TASKS = ["cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn"]

N_CLASSES = {"cars": 196, "dtd": 47, "eurosat": 10, "gtsrb": 43, "mnist": 10,
             "resisc45": 45, "sun397": 397, "svhn": 10}


class HFImageDataset(Dataset):
    def __init__(self, hf_name: str, split: str, transform=None,
                 image_key: str = "image", label_key: str = "label"):
        from datasets import load_dataset
        self.ds = load_dataset(hf_name, split=split,
                               cache_dir=str(DATA_ROOT / "hf_cache"))
        self.transform = transform
        self.image_key, self.label_key = image_key, label_key
        feat = self.ds.features[label_key]
        self.classes = getattr(feat, "names", None)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[int(i)]
        img = row[self.image_key].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row[self.label_key])


class SyntheticTask(Dataset):
    """Deterministic class-clustered images; learnable by a tiny ViT in a few
    steps. Class k = base noise + class-specific colored blob pattern."""

    def __init__(self, task_id: int, split: str, n_classes: int = 4,
                 n_per_class: int = 32, img_size: int = 64, transform=None):
        g = torch.Generator().manual_seed(1000 * task_id + {"train": 0, "val": 1,
                                                            "test": 2}[split])
        proto_g = torch.Generator().manual_seed(7 * task_id)  # split-invariant
        self.protos = torch.rand(n_classes, 3, img_size, img_size,
                                 generator=proto_g)
        self.x = []
        self.y = []
        for c in range(n_classes):
            noise = 0.35 * torch.randn(n_per_class, 3, img_size, img_size,
                                       generator=g)
            self.x.append((self.protos[c] + noise).clamp(0, 1))
            self.y += [c] * n_per_class
        self.x = torch.cat(self.x)
        self.y = torch.tensor(self.y)
        self.classes = [f"c{c}" for c in range(n_classes)]
        self.transform = None  # already tensors at target size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _trainval(full_train: Dataset, split: str, val_frac: float = 0.1):
    n = len(full_train)
    idx = np.random.RandomState(0).permutation(n)
    n_val = max(1, int(val_frac * n))
    chosen = idx[:n_val] if split == "val" else idx[n_val:]
    sub = Subset(full_train, chosen.tolist())
    sub.classes = getattr(full_train, "classes", None)
    return sub


def get_task(name: str, split: str, transform=None) -> Dataset:
    import torchvision.datasets as tvd
    root = str(DATA_ROOT)
    if name.startswith("toy"):
        return SyntheticTask(int(name[3:]), split, transform=transform)

    if name == "cars":
        hf_split = {"train": "train", "val": "train", "test": "test"}[split]
        ds = HFImageDataset("tanganke/stanford_cars", hf_split, transform)
        return _trainval(ds, split) if split != "test" else ds
    if name == "resisc45":
        hf_split = {"train": "train", "val": "train", "test": "test"}[split]
        ds = HFImageDataset("tanganke/resisc45", hf_split, transform)
        return _trainval(ds, split) if split != "test" else ds

    if name == "dtd":
        tv_split = {"train": "train", "val": "val", "test": "test"}[split]
        return tvd.DTD(root, split=tv_split, transform=transform, download=True)
    if name == "eurosat":
        full = tvd.EuroSAT(root, transform=transform, download=True)
        if split == "test":  # EuroSAT has no official split; carve 20% test
            n = len(full)
            idx = np.random.RandomState(1).permutation(n)
            sub = Subset(full, idx[: n // 5].tolist())
            sub.classes = full.classes
            return sub
        n = len(full)
        idx = np.random.RandomState(1).permutation(n)[n // 5:]
        train = Subset(full, idx.tolist())
        train.classes = full.classes
        return _trainval(train, split)
    if name == "gtsrb":
        if split == "test":
            return tvd.GTSRB(root, split="test", transform=transform, download=True)
        return _trainval(tvd.GTSRB(root, split="train", transform=transform,
                                   download=True), split)
    if name == "mnist":
        if split == "test":
            return tvd.MNIST(root, train=False, transform=transform, download=True)
        return _trainval(tvd.MNIST(root, train=True, transform=transform,
                                   download=True), split)
    if name == "sun397":
        # torchvision SUN397(download=True) pulls the Princeton tar, which is
        # dead (HTTP 404). Use the tanganke HF mirror like cars/resisc45; it
        # ships an official train/test split, so we drop the old 80/20 carve.
        hf_split = {"train": "train", "val": "train", "test": "test"}[split]
        ds = HFImageDataset("tanganke/sun397", hf_split, transform)
        return _trainval(ds, split) if split != "test" else ds
    if name == "svhn":
        if split == "test":
            return tvd.SVHN(root, split="test", transform=transform, download=True)
        return _trainval(tvd.SVHN(root, split="train", transform=transform,
                                  download=True), split)
    raise KeyError(f"unknown task {name!r}")


def n_classes(name: str) -> int:
    if name.startswith("toy"):
        return 4
    return N_CLASSES[name]


def make_transform(encoder, train: bool):
    """Standard transform pipeline matched to an encoder wrapper."""
    import torchvision.transforms as T
    size = encoder.img_size
    norm = T.Normalize(encoder.norm_mean, encoder.norm_std)
    to_rgb = T.Lambda(lambda im: im.convert("RGB") if hasattr(im, "convert") else im)
    if train:
        return T.Compose([to_rgb,
                          T.RandomResizedCrop(size, scale=(0.6, 1.0)),
                          T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    return T.Compose([to_rgb, T.Resize(int(size * 1.14)), T.CenterCrop(size),
                      T.ToTensor(), norm])
