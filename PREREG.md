# Pre-registration — Isotropy and LoRA Composability (DRAFT)

**Status: DRAFT.** This file is frozen and tagged (`prereg-v1`) at Gate 1
(end Week 3), after the pilot selects lambda_ft and merger hyperparameters.
Until the tag, entries marked PROPOSED may change; nothing else may.

## Hypotheses (committed now, plan section 0.1)

- **H1** SIGReg-LoRA (C1) increases post-merge retention vs vanilla LoRA (C0),
  holding backbone, tasks, merger, seeds fixed.
- **H2** Across Arm B backbones, retention is monotone increasing in
  *measured* backbone isotropy (possible saturation/reversal admitted; the
  x-axis is measured isotropy, never lambda).
- **H3** The effect is mediated by task-vector geometry: lower isotropy ⇒
  task updates concentrate in dominant backbone eigendirections ⇒ higher
  subspace overlap and sign-conflict rates ⇒ more interference.
- **H4** (stretch) Post-hoc ZCA whitening recovers part of the retention gap.

## Primary endpoint (committed)

Normalized retention R(S) = mean_i acc_merged(i)/acc_solo(i) under protocol
P1 (fixed-budget logistic probe re-fit on the frozen merged encoder:
cap 10,000 train samples, C=1.0, max_iter=1000, standardized features).
Secondary: worst-task retention, retention slope vs k, P2 (original heads,
appendix), predictive power of pre-merge geometric metrics.

## Statistical plan (committed)

- 3 fine-tuning seeds; bootstrap 95% CIs (10k resamples) over task x seed.
- H1: Wilcoxon signed-rank on per-task retention, C1 vs C0, paired within
  (task, merger, k, seed); Holm correction across the 5 mergers.
- H2: mixed-effects regression retention ~ isotropy + probe_acc +
  (1|task) + (1|merger); Spearman rho companion; isotonic-fit monotonicity.
- Powered for: 2-point retention gap (Arm A); visible ordering of >= 4 of 5
  backbones (Arm B). Below that, the framing shifts per Gate logic (plan §4).

## Frozen at prereg tag (currently PROPOSED in configs/grid.yaml)

- lambda_ft: winner of pilot sweep {0.02, 0.05, 0.1}.
- Merger hyperparameters: tuned once on the held-out validation pair
  (eurosat-resisc45, seed 0), then frozen for every experiment.
- k=2 pair list (5 close, 5 far), k=4 subsets (8, seed 7): configs/grid.yaml.
- Arm B dial: lambda_mult in {0, 0.1, 0.3, 1.0, 3.0} x repo default (0.05,
  to be confirmed against the LeJEPA paper recipe in Week 1 GPU
  verification BEFORE this file is tagged).

## Confound defense (committed)

lambda also shifts raw backbone quality. All three defenses reported:
(1) normalized retention divides out adapter quality; (2) IN-100 linear-probe
accuracy per backbone enters the regression as a covariate; (3) full per-task
pre-merge accuracy tables in the appendix.

## Flag-plant pilot deviations (preprint only; full study uses the above)

The priority-claim preprint runs a reduced protocol, stated honestly in the
paper's Limitations: single seed; pilot pairs only (eurosat-resisc45,
mnist-dtd); $\lambda_\text{ft}$ swept on the pilot itself (full study
re-selects on a held-out pair); and fine-tuning capped at
$\min(5\text{ epochs}, 800\text{ steps})$ -- the cap binds only on large
datasets (MNIST; RESISC45 if applicable), while DTD/EuroSAT run the full 5
epochs. Rationale: keep no single task's training length dominating a pilot;
MNIST is already accuracy-saturated well before 800 steps. The full campaign
removes the cap.

## Decision gates

Gate 1 (Week 3): proceed if C1-C0 gap >= 2 points on either pilot pair OR
the two mini-backbones order correctly with a visible gap; else pivot to the
"what does govern composability" autopsy. Gate 2 (Week 9): monotone /
sweet-spot / flat branching per plan §4. Gate 3 (Week 12): experiment freeze.
