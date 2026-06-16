"""Arm B cell: pretrain one LeJEPA ViT-S backbone at a given lambda multiple.

  python scripts/pretrain_lejepa.py --lambda-mult 1.0 --dataset imagenet100 \
      --epochs 200 --out results/pretrain/b3_lam1.0 [--done <path>]

Resume-safe: rerun the same command after a walltime kill.
"""

import argparse

from _common import ROOT, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-mult", type=float, required=True)
    ap.add_argument("--dataset", default="imagenet100",
                    choices=["imagenet100", "cifar100", "toy"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--n-views", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--model", default="vit_small_patch16_224")
    ap.add_argument("--done", default=None)
    args = ap.parse_args()

    from isomerge.pretrain import get_pretrain_dataset, pretrain_lejepa
    ds = get_pretrain_dataset(args.dataset, args.img_size, args.n_views)
    res = pretrain_lejepa(
        ds, ROOT / args.out, lambda_mult=args.lambda_mult,
        model_name=args.model, img_size=args.img_size, epochs=args.epochs,
        batch_size=args.batch_size, n_views=args.n_views, lr=args.lr,
        seed=args.seed, num_workers=args.num_workers, max_steps=args.max_steps)
    finish(args.done, {"steps": res["steps"], "out": res["out_dir"]})


if __name__ == "__main__":
    main()
