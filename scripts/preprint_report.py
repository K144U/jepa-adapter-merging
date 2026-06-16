"""Aggregate preprint-stage outputs into the paper's numbers.

  python scripts/preprint_report.py [--out results/reports/preprint]

Produces:
  preprint_rows.csv      one row per (backbone, merge set, condition, merger)
  h1_table.json          C1 vs C0 retention per merger + Wilcoxon/Holm
  h2_dose.json           per-backbone measured isotropy vs mean retention
  scatter.png            the centerpiece: isotropy vs retention, all backbones
  numbers.json           every \\TODO value referenced by paper/main.tex
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def load_merges():
    rows = []
    for p in sorted((ROOT / "results" / "merges").rglob("*.json")):
        r = json.loads(p.read_text())
        tag = p.parent.name
        if tag == "toy":
            continue
        cond = "c0"
        for c in (r.get("lambda_ft") or []):
            if c and c > 0:
                cond = f"c1lam{c}"
        rows.append({
            "backbone": tag, "merger": r["merger"], "k": r["k"],
            "set": "-".join(r["tasks"]), "cond": cond,
            "retention": r["retention_p1"]["retention_mean"],
            "retention_worst": r["retention_p1"]["retention_worst"],
            "acc_merged": r["retention_p1"]["acc_merged_mean"],
            "cos_mean": r["taskvec_geometry"]["pairwise_cosine_mean"],
            "sign_conflict": r["taskvec_geometry"]["sign_conflict_rate"],
            "subspace_overlap": r["taskvec_geometry"]["subspace_overlap_mean"],
        })
    return rows


def load_geometry():
    geo = {}
    for p in sorted((ROOT / "results" / "metrics").glob("geom_*_in100.json")):
        r = json.loads(p.read_text())
        tag = p.stem[len("geom_"):-len("_in100")]
        geo[tag] = r["pooled"]
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reports/preprint")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = load_merges()
    geo = load_geometry()
    if not rows:
        print("no merge results yet"); return

    import csv
    with open(out / "preprint_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ---- H1: C1 vs C0, paired within (set, merger), pilot backbone --------
    from isomerge.eval import wilcoxon_holm
    h1 = {}
    pilot_bb = sorted({r["backbone"] for r in rows
                       if any(x["backbone"] == r["backbone"]
                              and x["cond"].startswith("c1") for x in rows)})
    best_lam, best_gap = None, -1e9
    conds_c1 = sorted({r["cond"] for r in rows if r["cond"].startswith("c1")})
    for lam_cond in conds_c1:
        gaps = []
        for r in rows:
            if r["cond"] != lam_cond:
                continue
            base = [x for x in rows if x["backbone"] == r["backbone"]
                    and x["set"] == r["set"] and x["merger"] == r["merger"]
                    and x["cond"] == "c0"]
            if base:
                gaps.append(r["retention"] - base[0]["retention"])
        if gaps and np.mean(gaps) > best_gap:
            best_gap, best_lam = float(np.mean(gaps)), lam_cond
    if best_lam:
        paired = {}
        for merger in sorted({r["merger"] for r in rows}):
            c1 = [r["retention"] for r in rows
                  if r["cond"] == best_lam and r["merger"] == merger]
            c0 = [r["retention"] for r in rows
                  if r["cond"] == "c0" and r["merger"] == merger
                  and r["backbone"] in pilot_bb]
            if c1 and len(c1) == len(c0):
                paired[merger] = (c1, c0)
        h1 = {"best_lambda_cond": best_lam,
              "mean_gap_points": 100 * best_gap,
              "per_merger": wilcoxon_holm(paired) if paired else {}}
    (out / "h1_table.json").write_text(json.dumps(h1, indent=1))

    # ---- H2 + family scatter: isotropy vs retention ------------------------
    dose = {}
    for tag in sorted({r["backbone"] for r in rows}):
        if tag not in geo:
            continue
        rets = [r["retention"] for r in rows
                if r["backbone"] == tag and r["cond"] == "c0"]
        if rets:
            dose[tag] = {
                "isoscore": geo[tag]["isoscore"],
                "effective_rank_frac": geo[tag]["effective_rank_frac"],
                "retention_mean": float(np.mean(rets)),
                "retention_per_merger": {
                    m: float(np.mean([r["retention"] for r in rows
                                      if r["backbone"] == tag
                                      and r["cond"] == "c0"
                                      and r["merger"] == m]))
                    for m in sorted({r["merger"] for r in rows})},
                "n": len(rets)}
    (out / "h2_dose.json").write_text(json.dumps(dose, indent=1))

    rho = None
    if len(dose) >= 3:
        from scipy.stats import spearmanr
        iso = [v["isoscore"] for v in dose.values()]
        ret = [v["retention_mean"] for v in dose.values()]
        rho, p = spearmanr(iso, ret)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            for tag, v in dose.items():
                ax.scatter(v["isoscore"], v["retention_mean"], s=60)
                ax.annotate(tag, (v["isoscore"], v["retention_mean"]),
                            fontsize=8, xytext=(4, 4),
                            textcoords="offset points")
            ax.set_xlabel("measured isotropy (IsoScore, pooled embedding, IN-100)")
            ax.set_ylabel("normalized retention (mean over sets x mergers)")
            ax.set_title(f"Isotropy vs composability (Spearman rho={rho:.2f})")
            fig.tight_layout()
            fig.savefig(out / "scatter.png", dpi=200)
        except ImportError:
            print("matplotlib unavailable; scatter skipped")

    # ---- mechanism: geometry mediators vs isotropy --------------------------
    mech = {}
    for tag, v in dose.items():
        tr = [r for r in rows if r["backbone"] == tag and r["cond"] == "c0"]
        mech[tag] = {"isoscore": v["isoscore"],
                     "sign_conflict": float(np.mean([r["sign_conflict"] for r in tr])),
                     "subspace_overlap": float(np.mean([r["subspace_overlap"] for r in tr])),
                     "cos_mean": float(np.mean([r["cos_mean"] for r in tr]))}

    numbers = {"h1": h1, "h2_dose": dose, "mechanism": mech,
               "spearman_rho_family": rho, "n_rows": len(rows)}
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:2000])
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
