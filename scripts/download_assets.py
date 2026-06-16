"""Login-node download of everything the preprint scope needs (compute nodes
have no internet). Idempotent; rerun until it prints ALL OK.

Scope (flag-plant preprint): pilot tasks {dtd, eurosat, mnist, resisc45},
CIFAR-100 (LeJEPA smoke), ImageNet-100 (Arm B mini), and the three Arm A
ViT-L checkpoints (V-JEPA 2, MAE, DINOv2) into the project HF/timm cache.
"""

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

OK, BAD = [], []


def step(name, fn):
    try:
        fn()
        OK.append(name)
        print(f"  OK   {name}", flush=True)
    except Exception as e:  # noqa: BLE001
        BAD.append(name)
        print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=2)


def tv_datasets():
    import torchvision.datasets as tvd
    from isomerge.data import DATA_ROOT
    root = str(DATA_ROOT)
    tvd.DTD(root, split="train", download=True)
    tvd.DTD(root, split="val", download=True)
    tvd.DTD(root, split="test", download=True)
    tvd.EuroSAT(root, download=True)
    tvd.MNIST(root, train=True, download=True)
    tvd.MNIST(root, train=False, download=True)
    tvd.CIFAR100(root, train=True, download=True)


def hf_resisc45():
    from datasets import load_dataset
    from isomerge.data import DATA_ROOT
    for split in ["train", "test"]:
        load_dataset("tanganke/resisc45", split=split,
                     cache_dir=str(DATA_ROOT / "hf_cache"))


def hf_in100():
    from datasets import load_dataset
    from isomerge.data import DATA_ROOT
    for split in ["train", "validation"]:
        load_dataset("clane9/imagenet-100", split=split,
                     cache_dir=str(DATA_ROOT / "hf_cache"))


def ckpt_mae():
    import timm
    timm.create_model("vit_large_patch16_224.mae", pretrained=True,
                      num_classes=0)


def ckpt_dinov2():
    import timm
    timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True,
                      num_classes=0, img_size=224)


def ckpt_vjepa2():
    from transformers import AutoModel
    AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")


if __name__ == "__main__":
    step("torchvision datasets (dtd, eurosat, mnist, cifar100)", tv_datasets)
    step("HF resisc45", hf_resisc45)
    step("HF imagenet-100", hf_in100)
    step("checkpoint MAE ViT-L", ckpt_mae)
    step("checkpoint DINOv2 ViT-L", ckpt_dinov2)
    step("checkpoint V-JEPA 2 ViT-L", ckpt_vjepa2)
    print(("ALL OK" if not BAD else f"FAILED: {BAD}"), flush=True)
    sys.exit(1 if BAD else 0)
