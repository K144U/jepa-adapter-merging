"""Analysis for the pivoted preprint: 'Merging preserves features but breaks
heads on JEPA backbones, and an RD-optimal encoder fixes it.'

Produces the paper's three results objects from results/merges/*:
  T1  retention by merger x protocol (P1 re-fit vs P2 head-reuse), pooled
      over backbones+pairs -> the P1-ceiling / P2-interference localization.
  T2  P2 retention by backbone x merger -> rd_encoder dominance across the
      SSL families (V-JEPA 2, MAE, DINOv2, LeJEPA ViT-S) and two scales.
  T3  the P1-P2 gap per merger (the interference-localization statistic) with
      a paired Wilcoxon (P1 vs P2) and bootstrap CI on the rd_encoder margin.
Writes results/reports/pivot/{tables.json, summary.txt}.
"""

import json
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isomerge.eval import bootstrap_ci  # noqa: E402

BACKBONE_LABEL = {
    "vjepa2_l": "V-JEPA 2 ViT-L", "mae_l": "MAE ViT-L",
    "dinov2_l": "DINOv2 ViT-L", "lejepa_pilot_lam0.1": "LeJEPA ViT-S (low-reg)",
    "lejepa_pilot_lam1.0": "LeJEPA ViT-S (high-reg)",
}
MERGERS = ["uniform", "task_arithmetic", "ties", "dare_ties", "rd_encoder"]


def load():
    rows = []
    for p in glob.glob(str(ROOT / "results/merges/*/*.json")):
        bb = Path(p).parent.name
        if bb == "toy":
            continue
        r = json.load(open(p))
        # C0 only (vanilla LoRA) for the cross-method/backbone comparison
        if any(c and c > 0 for c in (r.get("lambda_ft") or [])):
            continue
        rows.append({"backbone": bb, "merger": r["merger"],
                     "pair": "-".join(r["tasks"]),
                     "p1": r["retention_p1"]["retention_mean"],
                     "p2": r["retention_p2"]["retention_mean"]})
    return rows


def main():
    rows = load()
    out = ROOT / "results/reports/pivot"
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no data yet"); return
    backbones = [b for b in BACKBONE_LABEL if any(r["backbone"] == b for r in rows)]
    lines = []

    def p(s=""):
        lines.append(s); print(s)

    # T1: merger x protocol, pooled
    p("=== T1: retention by merger x protocol (C0, pooled over backbones+pairs) ===")
    p(f"{'merger':16s} {'P1 (re-fit)':>14s} {'P2 (head reuse)':>16s} {'P1-P2 gap':>10s}")
    t1 = {}
    for m in MERGERS:
        p1 = [r["p1"] for r in rows if r["merger"] == m]
        p2 = [r["p2"] for r in rows if r["merger"] == m]
        if not p1:
            continue
        gap = np.mean(p1) - np.mean(p2)
        t1[m] = {"p1": float(np.mean(p1)), "p2": float(np.mean(p2)),
                 "gap": float(gap), "n": len(p1)}
        p(f"{m:16s} {np.mean(p1):14.3f} {np.mean(p2):16.3f} {gap:10.3f}")

    # T2: P2 by backbone x merger
    p("\n=== T2: P2 retention by backbone x merger (C0) ===")
    hdr = f"{'backbone':24s} " + " ".join(f"{m[:8]:>9s}" for m in MERGERS)
    p(hdr)
    t2 = {}
    for bb in backbones:
        cells = []
        t2[bb] = {}
        for m in MERGERS:
            v = [r["p2"] for r in rows if r["backbone"] == bb and r["merger"] == m]
            t2[bb][m] = float(np.mean(v)) if v else None
            cells.append(f"{np.mean(v):9.3f}" if v else f"{'--':>9s}")
        p(f"{BACKBONE_LABEL[bb]:24s} " + " ".join(cells))

    # T3: rd_encoder margin over best baseline on P2 (per backbone) + Wilcoxon
    p("\n=== T3: rd_encoder P2 margin over best averaging/TIES baseline ===")
    p(f"{'backbone':24s} {'rd_encoder':>10s} {'best base':>10s} {'margin':>8s}")
    t3 = {}
    for bb in backbones:
        rd = [r["p2"] for r in rows if r["backbone"] == bb and r["merger"] == "rd_encoder"]
        base = {m: np.mean([r["p2"] for r in rows
                            if r["backbone"] == bb and r["merger"] == m])
                for m in MERGERS if m != "rd_encoder"
                and any(r["backbone"] == bb and r["merger"] == m for r in rows)}
        if not rd or not base:
            continue
        best_m = max(base, key=base.get)
        margin = np.mean(rd) - base[best_m]
        t3[bb] = {"rd": float(np.mean(rd)), "best_baseline": best_m,
                  "best_baseline_val": float(base[best_m]), "margin": float(margin)}
        p(f"{BACKBONE_LABEL[bb]:24s} {np.mean(rd):10.3f} "
          f"{base[best_m]:10.3f} {margin:+8.3f}  (vs {best_m})")

    # overall P1-vs-P2 paired test + rd_encoder margin CI
    from scipy.stats import wilcoxon
    allp1 = [r["p1"] for r in rows]; allp2 = [r["p2"] for r in rows]
    try:
        w = float(wilcoxon(allp1, allp2).pvalue)
    except ValueError:
        w = float("nan")
    rd_margins = []
    for bb in backbones:
        for pair in set(r["pair"] for r in rows if r["backbone"] == bb):
            rd = [r["p2"] for r in rows if r["backbone"] == bb
                  and r["pair"] == pair and r["merger"] == "rd_encoder"]
            ta = [r["p2"] for r in rows if r["backbone"] == bb
                  and r["pair"] == pair and r["merger"] == "task_arithmetic"]
            if rd and ta:
                rd_margins.append(rd[0] - ta[0])
    ci = bootstrap_ci(rd_margins, n_boot=10000) if rd_margins else {}
    p("\n=== headline stats ===")
    p(f"P1 vs P2 paired Wilcoxon (all merges): p={w:.2e}  "
      f"(mean P1 {np.mean(allp1):.3f} vs P2 {np.mean(allp2):.3f})")
    if ci:
        p(f"rd_encoder - task_arithmetic on P2: {ci['mean']:+.3f} "
          f"[95% CI {ci['lo']:+.3f}, {ci['hi']:+.3f}], n={len(rd_margins)}")
    p(f"\nbackbones with data: {len(backbones)}/5  | total C0 merges: {len(rows)}")

    (out / "tables.json").write_text(json.dumps(
        {"T1": t1, "T2": t2, "T3": t3,
         "p1_p2_wilcoxon_p": w,
         "rd_vs_ta_p2_ci": ci, "n_backbones": len(backbones),
         "n_merges": len(rows)}, indent=1))
    (out / "summary.txt").write_text("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
