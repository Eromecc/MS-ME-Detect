> Historical note: this document predates or discusses experiments before the current Qwen14 segment fusion paper release. The current paper final model is `MS-ME-Detect Qwen14 segment fusion` with AUROC 0.907822 on `all_samples`, documented in `paper_release/README_paper_release.md`. Text-Koopman, Deep DMD, transition-only, and 0.6951/0.7120 rows in this file are historical/exploratory unless explicitly restated as the Qwen14 final model.

# Final Experiment Summary

## Baselines

The project evaluated basic statistical/structural features, full all-feature
models, cleanup variants, and feature ablations.  Full allfeatures improves
over the basic baseline, and ablations show that probability and scale-response
features carry useful signal.

## Source Generalization

Public benchmark splits show strong same-source and cross-source performance.
The external `all_samples` set is a stronger shifted target set.  The source
matrix and distribution-shift reports show that high public performance does not
guarantee robust external performance.

Key evidence:

- public same-source performance is high
- public cross-source performance remains strong for several train/test pairs
- `all_samples` is shifted relative to public sources
- Ghostbuster probability direction reverses on `all_samples`

## Transition-State Profiling

Transition-state profiling models token-loss trajectories as transitions among
train-only loss states.  The final selected method is:

```text
leave_out_ghostbuster + full_plus_1_5B_and_7B_transition
```

External `all_samples` result:

```text
AUROC 0.6951
AUPRC 0.6592
F1 0.6799
TPR@FPR5% 0.0933
ECE 0.1488
Brier 0.2459
```

This is the previous transition reference for the current release.

## Koopman / DMD-Lite

DMD-lite was implemented as a per-text closed-form DMD model over handcrafted
loss-sequence observables.  It provides independent spectral signal but is not
the best external method.  It is useful as a dynamics-inspired baseline.

## Deep DMD

Deep DMD was fully implemented and evaluated with 1.5B+7B sweeps and a
cross-source matrix.  It shows public-source signal, but the global-K and
pooled-z classifier formulation does not consistently outperform
transition-state profiling on the shifted `all_samples` target.

Deep DMD is therefore reported as a negative/diagnostic result rather than the
selected method.

## Strict Mathematical Text-Koopman

Strict mathematical Text-Koopman was implemented as a separate experiment:

```text
Qwen hidden-state token trajectory
  -> learned lifting hidden_size -> observable_dim
  -> per-document local truncated exact-DMD K_i
  -> spectral/residual scalar fingerprint
  -> classical classifier
```

Small validation facts:

```text
hidden_size = 1536
observable_dim = 3072
no PCA/random projection
no pooled-z classifier
no global shared K
per-document local K_i
spectral-only downstream classifier
leakage audit passed
```

The strict Text-Koopman loss-update ablation was completed after the original
strict-math validation. The original strict math training was not completely
reconstruction-only, but its DMD/multistep implementation was not explicit
enough and could silently zero dynamics terms on DMD failure. The updated
training objective exposes three loss modes:

```text
recon_only
recon_lin
recon_lin_multi
```

`L_lin` and `L_multi` are differentiable through the per-document local
`K_tilde_i` and back to `Z/g_theta`. The strict pipeline still has no global
shared `K`, no pooled-z classifier, and no label classification loss. The
leakage audit passed.

The full loss-update ablation produced a new best external `all_samples` row:

```text
feature_set = full_plus_transition_plus_strict_koopman
loss_mode = recon_only
dmd_rank = 16
AUROC 0.7120
AUPRC 0.6860
F1 0.6789
TPR@FPR5% 0.1400
ECE 0.1796
Brier 0.2569
```

The best low-FPR row was:

```text
loss_mode = recon_lin
dmd_rank = 32
AUROC 0.6577
AUPRC 0.6564
TPR@FPR5% 0.2067
```

This means the historical best external result within that experiment line was a strict Text-Koopman
loss-update combined-feature row, but the gain must be reported as an ablation
finding: the best overall row is `recon_only/rank16`, not `recon_lin_multi`.
`L_lin` helps low-FPR recall, while `L_multi` did not show an additional
benefit in this run.

## Final Selected Method

For the final result table, the strict Text-Koopman loss-update
`recon_only/rank16` combined-feature row was the historical best external within that experiment line
`all_samples` result. Transition-state profiling remains the previous best
reference and the practical baseline that the loss update compares against.

## Remaining Limitation

The main unresolved issue is low-FPR detection under external target shift.
Future work should prioritize target-like dataset construction, calibration,
and robust source/domain shift evaluation.
