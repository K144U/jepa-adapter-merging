"""Arm A/B cell: LoRA fine-tune one (encoder, task, condition, seed) and
record the solo P1 probe accuracy (the retention denominator).

  python scripts/finetune_lora.py --encoder mae_l --task dtd \
      --lambda-ft 0.05 --seed 0 --out results/adapters/mae_l/dtd_c1_s0.pt
"""

import argparse

from _common import ROOT, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lejepa-checkpoint", default=None)
    ap.add_argument("--task", required=True)
    ap.add_argument("--lambda-ft", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--whiten", default=None,
                    help="Arm E: path to a fitted ZCA .pt to wrap the encoder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", default=None)
    args = ap.parse_args()

    import torch
    from isomerge.adapt import finetune
    from isomerge.data import get_task, make_transform
    from isomerge.eval import extract_features, p1_probe
    from isomerge.models import build_encoder

    enc = build_encoder(args.encoder, args.lejepa_checkpoint)
    if args.whiten:
        from isomerge.models.whiten import WhitenedEncoder
        enc = WhitenedEncoder(enc, torch.load(ROOT / args.whiten,
                                              map_location="cpu",
                                              weights_only=False))
    out_path = ROOT / args.out
    res = finetune(enc, args.task, lambda_ft=args.lambda_ft, seed=args.seed,
                   r=args.r, alpha=args.alpha, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr,
                   max_steps=args.max_steps, num_workers=args.num_workers,
                   out_path=out_path)

    # Solo P1: probe accuracy of this task's own adapted encoder -- the
    # denominator of normalized retention under the primary protocol. Cap the
    # train extraction at the probe budget (random subset == extract-all-then-
    # subsample, but skips extracting features the probe would discard).
    from isomerge.eval.probe import P1_TRAIN_CAP
    import numpy as np
    from torch.utils.data import Subset
    tr = get_task(args.task, "train", make_transform(enc, train=False))
    if len(tr) > P1_TRAIN_CAP:
        idx = np.random.RandomState(0).permutation(len(tr))[:P1_TRAIN_CAP]
        _cls = getattr(tr, "classes", None)
        tr = Subset(tr, idx.tolist())
        tr.classes = _cls
    te = get_task(args.task, "test", make_transform(enc, train=False))
    Xtr, ytr = extract_features(enc, tr, num_workers=args.num_workers)
    Xte, yte = extract_features(enc, te, num_workers=args.num_workers)
    solo_p1 = p1_probe(Xtr, ytr, Xte, yte)
    print(f"[ft {args.task}] solo P1 {solo_p1:.4f}", flush=True)

    blob = torch.load(out_path, map_location="cpu", weights_only=False)
    blob["extra"]["solo_p1"] = solo_p1
    torch.save(blob, out_path)
    finish(args.done, {"test_acc": res["test_acc"], "solo_p1": solo_p1})


if __name__ == "__main__":
    main()
