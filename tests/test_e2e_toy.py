"""End-to-end pipeline check at toy scale on CPU (plan Week 1 deliverable:
'every component runs end-to-end on toy scale').

Generates the toy_e2e manifest, executes every cell sequentially exactly as
the orchestrator would (same commands, same done-file contract), then sanity-
checks the outputs: adapters exist, solo accuracy beats chance, all 5 mergers
produce retention reports, and the geometry/stats helpers consume them.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = str(ROOT / ".venv" / "bin" / "python")

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAIL.append(name)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                       text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    return r.returncode


# clean previous toy outputs so the run is a real test, not a done-file skip
for d in ["results/done/toy_e2e", "results/adapters/toy", "results/merges/toy"]:
    p = ROOT / d
    if p.exists():
        for f in sorted(p.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()

rc = sh(f"{PY} scripts/make_manifest.py --stage toy_e2e")
check("manifest generated", rc == 0)
cells = json.loads((ROOT / "configs/manifests/toy_e2e.json").read_text())
check("manifest cell count", len(cells) == 14, f"({len(cells)} cells)")

t0 = time.time()
for c in cells:
    rc = sh(c["cmd"])
    ok = rc == 0 and (ROOT / c["done"]).exists()
    check(f"cell {c['name']}", ok)
    if not ok:
        break
print(f"  ({time.time() - t0:.0f}s for {len(cells)} cells)")

# pretraining smoke: a few steps of the toy LeJEPA loop, then resume
out = ROOT / "results/pretrain/toy_smoke"
if out.exists():
    for f in sorted(out.rglob("*"), reverse=True):
        f.unlink()
rc = sh(f"{PY} scripts/pretrain_lejepa.py --lambda-mult 1.0 --dataset toy "
        f"--img-size 64 --model vit_tiny_patch16_224 --batch-size 32 "
        f"--n-views 2 --max-steps 6 --num-workers 0 --epochs 1 "
        f"--out results/pretrain/toy_smoke")
check("pretrain smoke", rc == 0 and (out / "encoder_final.pt").exists())
rc = sh(f"{PY} scripts/pretrain_lejepa.py --lambda-mult 1.0 --dataset toy "
        f"--img-size 64 --model vit_tiny_patch16_224 --batch-size 32 "
        f"--n-views 2 --max-steps 8 --num-workers 0 --epochs 1 "
        f"--out results/pretrain/toy_smoke")
check("pretrain resume", rc == 0)

# geometry profile on the toy encoder
rc = sh(f"{PY} scripts/geometry_profile.py --encoder toy --data toy0 "
        f"--n-images 128 --num-workers 0 "
        f"--out results/metrics/geom_toy_toy0.json")
check("geometry profile", rc == 0)
if rc == 0:
    geom = json.loads((ROOT / "results/metrics/geom_toy_toy0.json").read_text())
    check("geometry has per-block profiles",
          "block0" in geom and "pooled" in geom)

# inspect merge outputs: every merger reported retention for both tasks
merges = sorted((ROOT / "results/merges/toy").glob("*_c0_*.json"))
check("5 mergers reported (c0)", len(merges) == 5, f"({len(merges)})")
solo_ok, rets = True, {}
for m in merges:
    r = json.loads(m.read_text())
    rets[r["merger"]] = r["retention_p1"]["retention_mean"]
    if min(r["solo_p1"].values()) < 0.4:  # 4-class toy: chance = 0.25
        solo_ok = False
check("solo P1 beats chance on toy", solo_ok)
check("retentions finite and positive",
      all(0 < v < 2.5 for v in rets.values()), f"({rets})")
check("taskvec geometry attached",
      all("taskvec_geometry" in json.loads(m.read_text()) for m in merges))

# stats helpers consume the outputs
from isomerge.eval import bootstrap_ci, retention, wilcoxon_holm  # noqa: E402

ci = bootstrap_ci([rets[m] for m in rets], n_boot=200)
check("bootstrap CI sane", ci["lo"] <= ci["mean"] <= ci["hi"])
w = wilcoxon_holm({"m1": ([0.9, 0.8, 0.85, 0.95], [0.7, 0.6, 0.65, 0.75]),
                   "m2": ([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])})
check("wilcoxon+holm runs", w["m1"]["p_holm"] <= 1.0
      and w["m2"]["p_raw"] == 1.0)

print(f"\n{'ALL PASS' if not FAIL else f'FAILED: {FAIL}'}")
sys.exit(1 if FAIL else 0)
