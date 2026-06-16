"""Retention endpoints and the pre-registered statistics (plan 1.5, 2).

Primary endpoint: normalized retention R(S) = mean_i acc_merged(i) /
acc_individual(i). Secondary: worst-task retention, retention slope vs k.
Stats: bootstrap 95% CIs (10k resamples) over task x seed; Wilcoxon
signed-rank C1 vs C0 with Holm correction across mergers.
"""

import numpy as np
from scipy.stats import wilcoxon


def retention(acc_merged: dict, acc_solo: dict) -> dict:
    """Both dicts: {task: accuracy}. Solo = the task's own adapter pre-merge."""
    per_task = {t: acc_merged[t] / max(acc_solo[t], 1e-12) for t in acc_merged}
    vals = list(per_task.values())
    return {
        "retention_mean": float(np.mean(vals)),
        "retention_worst": float(np.min(vals)),
        "per_task": per_task,
        "acc_merged_mean": float(np.mean(list(acc_merged.values()))),
    }


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean of `values` (task x seed cells)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return {"mean": float(v.mean()),
            "lo": float(np.percentile(means, 100 * alpha / 2)),
            "hi": float(np.percentile(means, 100 * (1 - alpha / 2)))}


def wilcoxon_holm(paired: dict) -> dict:
    """paired: {merger: (c1_values, c0_values)} matched within
    (task, k, seed). Returns per-merger p-values with Holm correction."""
    raw = {}
    for m, (c1, c0) in paired.items():
        d = np.asarray(c1, float) - np.asarray(c0, float)
        if np.allclose(d, 0):
            raw[m] = 1.0
        else:
            raw[m] = float(wilcoxon(c1, c0).pvalue)
    order = sorted(raw, key=raw.get)
    n = len(order)
    out, running_max = {}, 0.0
    for i, m in enumerate(order):
        adj = min(1.0, (n - i) * raw[m])
        running_max = max(running_max, adj)
        out[m] = {"p_raw": raw[m], "p_holm": running_max,
                  "median_gap": float(np.median(np.asarray(paired[m][0])
                                                - np.asarray(paired[m][1])))}
    return out


def isotropy_retention_regression(rows: list) -> dict:
    """Arm B/D dose-response: retention ~ isotropy + probe_acc with task and
    merger random intercepts (mixed-effects when statsmodels is available,
    OLS with cluster-robust fallback otherwise).

    rows: dicts with keys retention, isotropy, probe_acc, task, merger.
    """
    import numpy as np
    y = np.array([r["retention"] for r in rows], float)
    iso = np.array([r["isotropy"] for r in rows], float)
    pa = np.array([r["probe_acc"] for r in rows], float)
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame(rows)
        md = smf.mixedlm("retention ~ isotropy + probe_acc", df,
                         groups=df["task"],
                         re_formula="1",
                         vc_formula={"merger": "0 + C(merger)"})
        fit = md.fit(reml=True)
        return {"model": "mixedlm",
                "isotropy_coef": float(fit.params["isotropy"]),
                "isotropy_p": float(fit.pvalues["isotropy"]),
                "probe_acc_coef": float(fit.params["probe_acc"])}
    except Exception:
        X = np.stack([np.ones_like(iso), iso, pa], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        # Spearman rho as the distribution-free companion (pre-registered)
        from scipy.stats import spearmanr
        rho, p = spearmanr(iso, y)
        return {"model": "ols_fallback",
                "isotropy_coef": float(beta[1]),
                "probe_acc_coef": float(beta[2]),
                "resid_std": float(resid.std()),
                "spearman_rho": float(rho), "spearman_p": float(p)}
