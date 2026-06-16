"""Generate orchestrator manifests from configs/grid.yaml.

  python scripts/make_manifest.py --stage <stage> [--out configs/manifests/]

Stages:
  pilot_pretrain   Arm B mini (2 lambdas, short schedule)        ~2 GPU-days
  pilot_arm_a      pilot fine-tunes: 2 pairs x {C0, C1 sweep} x 1 seed
  pilot_merge      pilot merges (2 mergers, both conditions)
  arm_b_pretrain   the 5-lambda sweep                            B1..B4 first, B0 last
  arm_a_ft         3 backbones x 8 tasks x 2 conditions x 3 seeds
  arm_b_ft         lejepa backbones x 8 tasks x C0 x 3 seeds
  merges           full merge grid (k=2,4,8 x 5 mergers) for every
                   (backbone, condition, seed) whose adapters exist
  geometry         isotropy profiles: every backbone x (in100 + 8 tasks)
  toy_e2e          CPU end-to-end pipeline check (also run by tests)

Python is invoked as .venv/bin/python so cells inherit the overlay env.
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = ".venv/bin/python"

# Flag-plant pilots cap fine-tuning at this many optimizer steps (upper bound
# on the 5-epoch schedule). Touches only large datasets (MNIST, and RESISC45
# if it exceeds it); DTD/EuroSAT run full 5 epochs under it. Protocol stated
# in PREREG.md. The full campaign (arm_a_ft/arm_b_ft) does NOT cap.
PREPRINT_FT_CAP = 800


def load_grid():
    return yaml.safe_load((ROOT / "configs" / "grid.yaml").read_text())


def cell(name, cmd, done, min_free_gb=20):
    return {"name": name, "cmd": f"{cmd} --done {done}", "done": done,
            "min_free_gb": min_free_gb}


def backbone_flags(tag):
    """tag = encoder key, lejepa_lam<mult> (full Arm B checkpoints) or
    lejepa_pilot_lam<mult> (short-schedule pilot checkpoints)."""
    if tag.startswith("lejepa_pilot_lam"):
        mult = tag[len("lejepa_pilot_lam"):]
        ck = f"results/pretrain/pilot_lam{mult}/encoder_final.pt"
        return f"--encoder lejepa_s --lejepa-checkpoint {ck}"
    if tag.startswith("lejepa_lam"):
        mult = tag[len("lejepa_lam"):]
        ck = f"results/pretrain/lam{mult}/encoder_final.pt"
        return f"--encoder lejepa_s --lejepa-checkpoint {ck}"
    return f"--encoder {tag}"


def adapter_path(tag, task, cond, seed):
    return f"results/adapters/{tag}/{task}_{cond}_s{seed}.pt"


def ft_cell(tag, task, cond, lam_ft, seed, lora, stage, extra=""):
    out = adapter_path(tag, task, cond, seed)
    done = f"results/done/{stage}/ft_{tag}_{task}_{cond}_s{seed}.done"
    cmd = (f"{PY} scripts/finetune_lora.py {backbone_flags(tag)} "
           f"--task {task} --lambda-ft {lam_ft} --seed {seed} "
           f"--r {lora['r']} --alpha {lora['alpha']} --epochs {lora['epochs']} "
           f"--batch-size {lora['batch_size']} --lr {lora['lr']} "
           f"--out {out} {extra}".strip())
    return cell(f"ft_{tag}_{task}_{cond}_s{seed}", cmd, done)


def merge_cell(tag, tasks, cond, seed, merger, margs, stage):
    adapters = ",".join(adapter_path(tag, t, cond, seed) for t in tasks)
    set_id = "-".join(tasks) if len(tasks) <= 4 else f"k{len(tasks)}_full"
    name = f"merge_{tag}_{set_id}_{cond}_s{seed}_{merger}"
    out = f"results/merges/{tag}/{set_id}_{cond}_s{seed}_{merger}.json"
    done = f"results/done/{stage}/{name}.done"
    kv = ",".join(f"{k}={v}" for k, v in margs.items())
    cmd = (f"{PY} scripts/merge_eval.py {backbone_flags(tag)} "
           f"--adapters {adapters} --merger {merger} "
           + (f"--merger-args {kv} " if kv else "") + f"--out {out}")
    return cell(name, cmd, done)


def merge_sets(g):
    """The frozen evaluation sets: k=2 (10 pairs), k=4 (8 subsets), k=8."""
    pairs = [tuple(p) for p in g["pairs_close"] + g["pairs_far"]]
    rng = random.Random(g["k4_seed"])
    k4, seen = [], set()
    while len(k4) < g["k4_subsets"]:
        s = tuple(sorted(rng.sample(g["tasks"], 4)))
        if s not in seen:
            seen.add(s)
            k4.append(s)
    return pairs + k4 + [tuple(g["tasks"])]


def build(stage):
    g = load_grid()
    lora = g["lora"]
    cells = []

    if stage == "pilot_pretrain":
        for m in g["arm_b"]["pilot_lambda_mults"]:
            cells.append(cell(
                f"pretrain_pilot_lam{m}",
                f"{PY} scripts/pretrain_lejepa.py --lambda-mult {m} "
                f"--dataset {g['arm_b']['dataset']} "
                f"--epochs {g['arm_b']['pilot_epochs']} "
                f"--batch-size {g['arm_b']['batch_size']} "
                f"--out results/pretrain/pilot_lam{m}",
                f"results/done/pilot_pretrain/lam{m}.done", min_free_gb=30))

    elif stage == "arm_b_pretrain":
        mults = sorted(g["arm_b"]["lambda_mults"], reverse=True)  # B0 last
        for m in mults:
            cells.append(cell(
                f"pretrain_lam{m}",
                f"{PY} scripts/pretrain_lejepa.py --lambda-mult {m} "
                f"--dataset {g['arm_b']['dataset']} "
                f"--epochs {g['arm_b']['epochs']} "
                f"--batch-size {g['arm_b']['batch_size']} "
                f"--out results/pretrain/lam{m}",
                f"results/done/arm_b_pretrain/lam{m}.done", min_free_gb=30))

    elif stage == "pilot_arm_a":
        p = g["pilot"]
        tag = p["backbone"]
        tasks = sorted({t for pair in p["pairs"] for t in pair})
        for task in tasks:
            cells.append(ft_cell(tag, task, "c0", 0.0, 0, lora, stage))
            for lam in p["lambda_ft_sweep"]:
                cells.append(ft_cell(tag, task, f"c1lam{lam}", lam, 0, lora,
                                     stage))

    elif stage == "pilot_merge":
        p = g["pilot"]
        tag = p["backbone"]
        conds = ["c0"] + [f"c1lam{l}" for l in p["lambda_ft_sweep"]]
        for pair, cond, merger in itertools.product(p["pairs"], conds,
                                                    p["mergers"]):
            cells.append(merge_cell(tag, tuple(pair), cond, 0, merger,
                                    g["mergers"][merger], stage))

    elif stage == "arm_a_ft":
        for tag, task, (cond, cd), seed in itertools.product(
                g["arm_a_backbones"], g["tasks"], g["conditions"].items(),
                g["seeds"]):
            cells.append(ft_cell(tag, task, cond, cd["lambda_ft"], seed,
                                 lora, stage))

    elif stage == "arm_b_ft":
        tags = [f"lejepa_lam{m}" for m in g["arm_b"]["lambda_mults"]]
        for tag, task, seed in itertools.product(tags, g["tasks"], g["seeds"]):
            cells.append(ft_cell(tag, task, "c0", 0.0, seed, lora, stage))

    elif stage == "merges":
        tags = (list(g["arm_a_backbones"])
                + [f"lejepa_lam{m}" for m in g["arm_b"]["lambda_mults"]])
        sets_ = merge_sets(g)
        for tag in tags:
            conds = (list(g["conditions"]) if tag in g["arm_a_backbones"]
                     else ["c0"])
            for s, cond, seed, (merger, margs) in itertools.product(
                    sets_, conds, g["seeds"], g["mergers"].items()):
                # only emit cells whose adapters exist (stage-gated)
                if all((ROOT / adapter_path(tag, t, cond, seed)).exists()
                       for t in s):
                    cells.append(merge_cell(tag, s, cond, seed, merger,
                                            margs, stage))

    elif stage == "geometry":
        tags = (list(g["arm_a_backbones"])
                + [f"lejepa_lam{m}" for m in g["arm_b"]["lambda_mults"]])
        for tag in tags:
            for data in ["in100"] + g["tasks"]:
                name = f"geom_{tag}_{data}"
                cells.append(cell(
                    name,
                    f"{PY} scripts/geometry_profile.py {backbone_flags(tag)} "
                    f"--data {data} --out results/metrics/{name}.json",
                    f"results/done/geometry/{name}.done", min_free_gb=10))

    # ---- flag-plant preprint stages (sequential manifests: cells are
    # emitted in dependency order and the pilot runs on ONE GPU, so the
    # single FIFO worker preserves ordering; do not run these on >1 GPU) ----
    elif stage == "week1_verify":
        # batch-32 verification needs ~15GB; gate at 20 so a partially
        # occupied shared GPU still admits it (ft gates are set from the
        # VRAM numbers this cell reports)
        cells.append(cell(
            "week1_verify",
            f"{PY} scripts/verify_gpu.py",
            "results/done/week1/verify.done", min_free_gb=20))

    elif stage == "preprint_arm_a":
        p, pp = g["pilot"], g["preprint"]
        tag = p["backbone"]
        lora_p = dict(lora, epochs=pp["lora_epochs"])
        conds = [("c0", 0.0)] + [(f"c1lam{l}", l) for l in p["lambda_ft_sweep"]]
        for task, (cond, lam) in itertools.product(pp["tasks"], conds):
            c = ft_cell(tag, task, cond, lam, 0, lora_p, stage,
                        extra=f"--max-steps {PREPRINT_FT_CAP}")
            c["min_free_gb"] = 35
            cells.append(c)
        for pair, (cond, _), (merger, margs) in itertools.product(
                pp["pairs"], conds, g["mergers"].items()):
            cells.append(merge_cell(tag, tuple(pair), cond, 0, merger,
                                    margs, stage))
        for data in ["in100"] + pp["tasks"]:
            name = f"geom_{tag}_{data}"
            cells.append(cell(
                name,
                f"{PY} scripts/geometry_profile.py {backbone_flags(tag)} "
                f"--data {data} --out results/metrics/{name}.json",
                f"results/done/{stage}/{name}.done", min_free_gb=15))

    elif stage == "preprint_arm_b":
        pp = g["preprint"]
        lora_p = dict(lora, epochs=pp["lora_epochs"])
        for m in g["arm_b"]["pilot_lambda_mults"]:
            cells.append(cell(
                f"pretrain_pilot_lam{m}",
                f"{PY} scripts/pretrain_lejepa.py --lambda-mult {m} "
                f"--dataset {g['arm_b']['dataset']} "
                f"--epochs {g['arm_b']['pilot_epochs']} "
                f"--batch-size {g['arm_b']['batch_size']} "
                f"--out results/pretrain/pilot_lam{m}",
                f"results/done/pilot_pretrain/lam{m}.done", min_free_gb=30))
        tags = [f"lejepa_pilot_lam{m}" for m in g["arm_b"]["pilot_lambda_mults"]]
        for tag, task in itertools.product(tags, pp["tasks"]):
            cells.append(ft_cell(tag, task, "c0", 0.0, 0, lora_p, stage,
                                 extra=f"--max-steps {PREPRINT_FT_CAP}"))
        for tag, pair, (merger, margs) in itertools.product(
                tags, pp["pairs"], g["mergers"].items()):
            cells.append(merge_cell(tag, tuple(pair), "c0", 0, merger,
                                    margs, stage))
        for tag, data in itertools.product(tags, ["in100"] + pp["tasks"]):
            name = f"geom_{tag}_{data}"
            cells.append(cell(
                name,
                f"{PY} scripts/geometry_profile.py {backbone_flags(tag)} "
                f"--data {data} --out results/metrics/{name}.json",
                f"results/done/{stage}/{name}.done", min_free_gb=15))

    elif stage == "preprint_family":
        pp = g["preprint"]
        lora_p = dict(lora, epochs=pp["lora_epochs"])
        for tag, task in itertools.product(pp["family_backbones"], pp["tasks"]):
            c = ft_cell(tag, task, "c0", 0.0, 0, lora_p, stage,
                        extra=f"--max-steps {PREPRINT_FT_CAP}")
            c["min_free_gb"] = 35
            cells.append(c)
        for tag, pair, (merger, margs) in itertools.product(
                pp["family_backbones"], pp["pairs"], g["mergers"].items()):
            cells.append(merge_cell(tag, tuple(pair), "c0", 0, merger,
                                    margs, stage))
        for tag, data in itertools.product(pp["family_backbones"],
                                           ["in100"] + pp["tasks"]):
            name = f"geom_{tag}_{data}"
            cells.append(cell(
                name,
                f"{PY} scripts/geometry_profile.py {backbone_flags(tag)} "
                f"--data {data} --out results/metrics/{name}.json",
                f"results/done/{stage}/{name}.done", min_free_gb=15))

    elif stage == "toy_e2e":
        lora_toy = dict(lora, epochs=1, batch_size=32, lr="3e-3")
        for task in ["toy0", "toy1"]:
            for cond, lam in [("c0", 0.0), ("c1", 0.05)]:
                c = ft_cell("toy", task, cond, lam, 0, lora_toy, stage,
                            extra="--max-steps 30 --num-workers 0")
                c["min_free_gb"] = 0
                cells.append(c)
        for cond in ["c0", "c1"]:
            for merger, margs in load_grid()["mergers"].items():
                c = merge_cell("toy", ("toy0", "toy1"), cond, 0, merger,
                               margs, stage)
                c["cmd"] = c["cmd"].replace("--merger", "--num-workers 0 --merger")
                c["min_free_gb"] = 0
                cells.append(c)
    else:
        raise SystemExit(f"unknown stage {stage}")

    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default="configs/manifests")
    args = ap.parse_args()
    cells = build(args.stage)
    out = ROOT / args.out / f"{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells, indent=1))
    print(f"{out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
