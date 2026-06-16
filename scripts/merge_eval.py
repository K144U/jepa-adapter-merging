"""Merge cell: merge a set of adapters with one merger, evaluate P1 + P2 on
every task in the set, compute retention, and run the Arm C task-vector
geometry + functional-interference metrics on the way.

  python scripts/merge_eval.py --encoder mae_l --merger ties \
      --adapters results/adapters/mae_l/dtd_c0_s0.pt,results/adapters/... \
      --out results/merges/mae_l/k2_dtd-eurosat_c0_s0_ties.json

Adapter filenames carry their task in extra['task']; merger hyperparameters
come from --merger-args 'k=v,k=v' (frozen via the manifest, never tuned
per-result).
"""

import argparse

from _common import ROOT, finish


def parse_kv(s):
    out = {}
    if s:
        for part in s.split(","):
            k, v = part.split("=")
            try:
                out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lejepa-checkpoint", default=None)
    ap.add_argument("--adapters", required=True,
                    help="comma-separated adapter .pt paths")
    ap.add_argument("--merger", required=True)
    ap.add_argument("--merger-args", default="")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-probe-n", type=int, default=None)
    ap.add_argument("--whiten", default=None,
                    help="Arm E: ZCA .pt; must match the one used at FT time")
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", default=None)
    args = ap.parse_args()

    import torch
    from isomerge.adapt.lora import apply_task_vector, task_vector
    from isomerge.data import get_task, make_transform
    from isomerge.eval import extract_features, p1_probe, p2_head, retention
    from isomerge.merging import MERGERS
    from isomerge.metrics import feature_drift, linear_cka, taskvec_summary
    from isomerge.models import build_encoder
    from isomerge.utils import device, save_json

    paths = [ROOT / p for p in args.adapters.split(",")]
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    tasks = [b["extra"]["task"] for b in blobs]
    tvs = [task_vector(b["lora"]) for b in blobs]
    solo_p1 = {t: b["extra"]["solo_p1"] for t, b in zip(tasks, blobs)}
    solo_p2 = {t: b["extra"]["test_acc"] for t, b in zip(tasks, blobs)}

    geom = taskvec_summary(tvs, names=tasks)
    merged_tv = MERGERS[args.merger](tvs, **parse_kv(args.merger_args))

    def fresh_encoder():
        e = build_encoder(args.encoder, args.lejepa_checkpoint)
        if args.whiten:
            from isomerge.models.whiten import WhitenedEncoder
            e = WhitenedEncoder(e, torch.load(ROOT / args.whiten,
                                              map_location="cpu",
                                              weights_only=False))
        return e.to(device())

    enc = fresh_encoder()
    apply_task_vector(enc, merged_tv)

    from isomerge.eval.probe import P1_TRAIN_CAP

    def cap_train(ds, cap=P1_TRAIN_CAP, seed=0):
        # The P1 probe fits on at most P1_TRAIN_CAP samples (random subset).
        # Extracting the full train set and discarding the rest is pure waste,
        # so draw the random subset at the dataset level and extract only it.
        # Statistically identical to extract-all-then-subsample.
        import numpy as np
        from torch.utils.data import Subset
        n = len(ds)
        if n <= cap:
            return ds
        idx = np.random.RandomState(seed).permutation(n)[:cap]
        sub = Subset(ds, idx.tolist())
        sub.classes = getattr(ds, "classes", None)
        return sub

    acc_p1, acc_p2, drift, cka = {}, {}, {}, {}
    for t, b in zip(tasks, blobs):
        tr = cap_train(get_task(t, "train", make_transform(enc, train=False)))
        te = get_task(t, "test", make_transform(enc, train=False))
        Xtr, ytr = extract_features(enc, tr, num_workers=args.num_workers,
                                    max_n=args.max_probe_n)
        Xte, yte = extract_features(enc, te, num_workers=args.num_workers,
                                    max_n=args.max_probe_n)
        acc_p1[t] = p1_probe(Xtr, ytr, Xte, yte)
        acc_p2[t] = p2_head(b["extra"]["head_w"], b["extra"]["head_b"], Xte, yte)

        # functional interference vs the task's own solo encoder
        solo = fresh_encoder()
        apply_task_vector(solo, task_vector(b["lora"]))
        Xte_solo, _ = extract_features(solo, te, num_workers=args.num_workers,
                                       max_n=args.max_probe_n)
        drift[t] = feature_drift(Xte, Xte_solo)
        cka[t] = linear_cka(Xte, Xte_solo)
        del solo

    result = {
        "encoder": args.encoder, "merger": args.merger,
        "merger_args": parse_kv(args.merger_args),
        "tasks": tasks, "k": len(tasks),
        "adapters": [str(p) for p in paths],
        "acc_p1_merged": acc_p1, "acc_p2_merged": acc_p2,
        "solo_p1": solo_p1, "solo_p2": solo_p2,
        "retention_p1": retention(acc_p1, solo_p1),
        "retention_p2": retention(acc_p2, solo_p2),
        "feature_drift": drift, "cka": cka,
        "taskvec_geometry": geom,
        "seeds": [b["extra"].get("seed") for b in blobs],
        "lambda_ft": [b["extra"].get("lambda_ft") for b in blobs],
    }
    save_json(result, ROOT / args.out)
    print(f"[merge {args.merger} k={len(tasks)}] "
          f"R_p1 {result['retention_p1']['retention_mean']:.4f} "
          f"worst {result['retention_p1']['retention_worst']:.4f}", flush=True)
    finish(args.done, {"retention_p1": result["retention_p1"]["retention_mean"]})


if __name__ == "__main__":
    main()
