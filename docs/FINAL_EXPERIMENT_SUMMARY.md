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

This remains the main model for the current release.

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

The spectral-only strict model reaches roughly AUROC 0.52-0.54 on
`all_samples` in small validation.  Combining full/transition features with
strict Text-Koopman improves TPR@FPR5% in small validation, but does not beat
the transition-state profiling main model on AUROC/AUPRC.

## Final Selected Method

Transition-state profiling remains the selected main method for this release.

## Remaining Limitation

The main unresolved issue is low-FPR detection under external target shift.
Future work should prioritize target-like dataset construction, calibration,
and robust source/domain shift evaluation.
