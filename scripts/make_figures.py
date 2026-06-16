"""Paper figures from results/merges/* -> paper/fig_*.pdf (+ .png).

fig_localization: (a) P1 vs P2 retention by merger, pooled, showing the
interference-localization gap; (b) P2 retention by backbone x merger, showing
RD-encoder best on 4/5 backbones.
"""

import json
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MERGERS = ["uniform", "task_arithmetic", "ties", "dare_ties", "rd_encoder"]
MLABEL = {"uniform": "Uniform", "task_arithmetic": "Task Arith.",
          "ties": "TIES", "dare_ties": "DARE-TIES", "rd_encoder": "RD-encoder"}
BB = {"vjepa2_l": "V-JEPA 2", "mae_l": "MAE", "dinov2_l": "DINOv2",
      "lejepa_pilot_lam0.1": "LeJEPA-S (lo)", "lejepa_pilot_lam1.0": "LeJEPA-S (hi)"}


def load():
    P1 = defaultdict(list); P2 = defaultdict(list); byb = defaultdict(dict)
    for p in glob.glob(str(ROOT / "results/merges/*/*.json")):
        bb = Path(p).parent.name
        if bb == "toy":
            continue
        r = json.load(open(p))
        if any(c and c > 0 for c in (r.get("lambda_ft") or [])):
            continue
        P1[r["merger"]].append(r["retention_p1"]["retention_mean"])
        P2[r["merger"]].append(r["retention_p2"]["retention_mean"])
        byb[bb].setdefault(r["merger"], []).append(
            r["retention_p2"]["retention_mean"])
    return P1, P2, byb


def main():
    P1, P2, byb = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) P1 vs P2 by merger
    x = np.arange(len(MERGERS)); w = 0.38
    p1m = [np.mean(P1[m]) for m in MERGERS]
    p2m = [np.mean(P2[m]) for m in MERGERS]
    ax1.bar(x - w/2, p1m, w, label="P1 (re-fit probe)", color="#4C72B0")
    ax1.bar(x + w/2, p2m, w, label="P2 (head reuse)", color="#C44E52")
    ax1.axhline(1.0, ls=":", c="gray", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels([MLABEL[m] for m in MERGERS],
                                           rotation=25, ha="right")
    ax1.set_ylabel("normalized retention $R(S)$")
    ax1.set_title("(a) Interference lives in the head (pooled)")
    ax1.set_ylim(0, 1.35); ax1.legend(loc="upper right", fontsize=9)
    for xi, (a, b) in enumerate(zip(p1m, p2m)):
        ax1.text(xi, max(a, b) + 0.02, f"$\\Delta${a-b:+.2f}", ha="center",
                 fontsize=7.5, color="dimgray")

    # (b) P2 by backbone x merger
    backbones = [b for b in BB if b in byb]
    xb = np.arange(len(backbones)); ww = 0.16
    for i, m in enumerate(MERGERS):
        vals = [np.mean(byb[b].get(m, [np.nan])) for b in backbones]
        ax2.bar(xb + (i - 2) * ww, vals, ww, label=MLABEL[m])
    ax2.set_xticks(xb); ax2.set_xticklabels([BB[b] for b in backbones],
                                            rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("P2 retention")
    ax2.set_title("(b) RD-encoder best on 4/5 backbones")
    ax2.set_ylim(0, 1.05); ax2.legend(ncol=2, fontsize=7.5, loc="lower center")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"paper/fig_localization.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote paper/fig_localization.{pdf,png}")


if __name__ == "__main__":
    main()
