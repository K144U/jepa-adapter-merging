"""Week 1 GPU verification (plan section 8, item 2) -- the only step that can
invalidate the design. One cell on one GPU, fully offline (assets must be
pre-downloaded by scripts/download_assets.py on the login node).

Per backbone {vjepa2_l, mae_l, dinov2_l}: load, forward a real EuroSAT batch,
inject LoRA, take 3 optimization steps (loss must decrease or stay finite),
report VRAM and step time. V-JEPA 2 runs the single-frame-replication path --
its result decides the plan 1.1 fallback question. Writes
results/metrics/week1_verify.json.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from _common import finish  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="vjepa2_l,mae_l,dinov2_l")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/metrics/week1_verify.json")
    ap.add_argument("--done", default=None)
    args = ap.parse_args()

    import torch
    from isomerge.adapt.lora import inject_lora
    from isomerge.adapt.sigreg import SIGReg
    from isomerge.data import get_task, make_transform
    from isomerge.metrics import isotropy_profile
    from isomerge.models import build_encoder
    from isomerge.utils import device, save_json

    dev = device()
    assert dev.type == "cuda", "verification must run on a GPU"
    report = {"gpu": torch.cuda.get_device_name(0)}

    for key in args.backbones.split(","):
        rec = {}
        try:
            t0 = time.time()
            enc = build_encoder(key).to(dev)
            rec["load_s"] = round(time.time() - t0, 1)
            rec["embed_dim"] = enc.embed_dim
            rec["n_lora_targets"] = len(enc.lora_targets())

            ds = get_task("eurosat", "train", make_transform(enc, train=True))
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4)
            x, y = next(iter(loader))
            x, y = x.to(dev), y.to(dev)

            enc.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                z = enc(x)
            rec["forward_shape"] = list(z.shape)
            rec["embedding_isotropy"] = isotropy_profile(z.float())

            wrapped = inject_lora(enc, enc.lora_targets())
            head = torch.nn.Linear(enc.embed_dim, 10).to(dev)
            sig = SIGReg().to(dev)
            params = [p for m in wrapped.values()
                      for p in (m.lora_A, m.lora_B)] + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=1e-4)
            enc.train()
            losses = []
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(3):
                with torch.autocast("cuda", torch.bfloat16):
                    zz = enc(x)
                    loss = (torch.nn.functional.cross_entropy(head(zz), y)
                            + 0.05 * sig(zz.float()))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss))
            rec["step_s"] = round((time.time() - t0) / 3, 2)
            rec["losses"] = [round(l, 4) for l in losses]
            rec["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            rec["ok"] = all(map(lambda v: v == v, losses))  # finite
            del enc, wrapped, head, opt
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            torch.cuda.empty_cache()
        report[key] = rec
        print(f"[verify] {key}: {rec}", flush=True)

    save_json(report, ROOT / args.out)
    all_ok = all(v.get("ok") for k, v in report.items() if k != "gpu")
    print(f"[verify] {'ALL OK' if all_ok else 'SOME FAILED'}", flush=True)
    if all_ok:
        finish(args.done, report)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
