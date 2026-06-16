"""Arm E cell: fit ZCA whitening of an encoder's pooled-embedding space.

  python scripts/fit_whitening.py --encoder lejepa_s \
      --lejepa-checkpoint results/pretrain/lam0.1/encoder_final.pt \
      --data in100 --out results/metrics/zca_lam0.1.pt
"""

import argparse

from _common import ROOT, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lejepa-checkpoint", default=None)
    ap.add_argument("--data", default="in100")
    ap.add_argument("--n-images", type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", default=None)
    args = ap.parse_args()

    import torch
    from isomerge.data import get_task, make_transform
    from isomerge.eval import extract_features
    from isomerge.models import build_encoder
    from isomerge.models.whiten import fit_zca
    from isomerge.utils import device

    enc = build_encoder(args.encoder, args.lejepa_checkpoint).to(device())
    if args.data == "in100":
        from isomerge.data.tasks import HFImageDataset
        ds = HFImageDataset("clane9/imagenet-100", "validation",
                            make_transform(enc, train=False))
    else:
        ds = get_task(args.data, "val", make_transform(enc, train=False))
    X, _ = extract_features(enc, ds, num_workers=args.num_workers,
                            max_n=args.n_images)
    zca = fit_zca(X)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(zca, out)
    print(f"[zca {args.encoder}] fit on {len(X)} features -> {out}", flush=True)
    finish(args.done)


if __name__ == "__main__":
    main()
