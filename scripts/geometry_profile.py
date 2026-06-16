"""Backbone-geometry cell: isotropy profile of an encoder, per layer and at
the pooled embedding, on a reference dataset and/or task data (plan 1.3).

  python scripts/geometry_profile.py --encoder mae_l --data dtd \
      --out results/metrics/geometry_mae_l_dtd.json

--data accepts a task name (val split), 'toy0'.., or 'in100' (the IN-100
validation set, the backbone-intrinsic measurement for Arm B).
"""

import argparse

from _common import ROOT, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lejepa-checkpoint", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-images", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", default=None)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from isomerge.data import get_task, make_transform
    from isomerge.metrics import isotropy_profile
    from isomerge.models import build_encoder
    from isomerge.utils import device, save_json

    enc = build_encoder(args.encoder, args.lejepa_checkpoint).to(device())
    enc.eval()
    if args.data == "in100":
        from isomerge.data.tasks import HFImageDataset
        ds = HFImageDataset("clane9/imagenet-100", "validation",
                            make_transform(enc, train=False))
    else:
        ds = get_task(args.data, "val", make_transform(enc, train=False))

    # hook every transformer block: token-mean output per layer
    feats = {}
    blocks = (enc.net.blocks if hasattr(enc, "net") and hasattr(enc.net, "blocks")
              else None)
    hooks = []
    if blocks is not None:
        def mk(i):
            def hook(_m, _inp, out):
                o = out[0] if isinstance(out, tuple) else out
                feats.setdefault(f"block{i}", []).append(
                    o.mean(dim=1).float().cpu())
            return hook
        hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(blocks)]

    pooled = []
    n = 0
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers)
    with torch.no_grad():
        for x, _ in loader:
            pooled.append(enc(x.to(device())).float().cpu())
            n += len(x)
            if n >= args.n_images:
                break
    for h in hooks:
        h.remove()

    out = {"encoder": args.encoder, "data": args.data, "n_images": n,
           "pooled": isotropy_profile(torch.cat(pooled))}
    for k, v in feats.items():
        out[k] = isotropy_profile(torch.cat(v))
    save_json(out, ROOT / args.out)
    print(f"[geometry {args.encoder}/{args.data}] "
          f"erank {out['pooled']['effective_rank']:.1f} "
          f"isoscore {out['pooled']['isoscore']:.4f}", flush=True)
    finish(args.done, {"pooled": out["pooled"]})


if __name__ == "__main__":
    main()
