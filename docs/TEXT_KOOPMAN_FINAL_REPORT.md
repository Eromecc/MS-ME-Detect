# Strict Text-Koopman Final Report

Report timestamp: 2026-05-17 UTC.

## Implementation Status

The strict Text-Koopman spectral-only pipeline has been implemented under new
output roots only:

- `features_hidden_states/`
- `features_text_koopman/`
- `checkpoints_text_koopman/`
- `results_text_koopman/`

No existing Deep DMD, DMD-lite, transition, or scale-response result directory
was intentionally overwritten.

The implemented path is:

```text
Qwen hidden-state trajectory
  -> train-only projection
  -> learned lifting g_theta
  -> per-document local K_i
  -> local-K spectral/residual scalar summaries
  -> classical downstream classifier
```

## Required Report Answers

1. Hidden-state cache extraction:

Qwen2.5-1.5B hidden-state cache exists for `m4_train/dev/test`,
`ghostbuster_train/dev/test`, `hc3_plus_train/dev/test`, and `all_samples`.
The current manifests report `hidden_size=1536`, `max_length=256`,
`dtype=float16`, and `failed_ids=[]`.

2. `hidden_size` source:

`src/hidden_state_cache.py` reads `model.config.hidden_size`; it does not
hard-code 4096.

3. Learned lifting and dimension constraint:

`TextKoopmanLifting` enforces `observable_dim > projection_dim`. Supported
projectors are `random`, `pca`, and `linear`. Linear mode trains the
hidden_size -> projection_dim layer inside `g_theta`.

4. Local `K_i`:

Feature extraction estimates a reduced local DMD operator per document via
`local_dmd_reduced`. The lifting model has no global shared `K` parameter.

5. Downstream classifier inputs:

The strict spectral columns have prefix `text_koopman_`; sampled feature CSVs
contain only `id` plus `text_koopman_*` columns. The latest current
`leave_out_ghostbuster/proj256/rank32` run used two source-guarded spectral
features and no banned feature names.

6. Leakage audit:

Latest current run:

```text
leakage_audit.passed=true
banned_feature_names=[]
source_artifact_risk=false
label_probe_dev_accuracy=0.5921
source_probe_dev_accuracy=0.4429
domain_probe_dev_accuracy=0.3936
```

This pass uses train/dev-only source guarding. Earlier full spectral feature
sets showed source-artifact risk, so unguarded spectral results should not be
treated as source/domain invariant.

7. `text_koopman_spectral_only` public/all_samples result:

Latest current `all_samples`:

```text
AUROC 0.4892
AUPRC 0.4838
F1 0.6480
TPR@FPR1% 0.0000
TPR@FPR5% 0.0000
FPR@TPR95% 0.9400
MCC -0.0131
precision 0.4982
recall 0.9267
accuracy 0.4967
Brier 0.3858
ECE 0.3419
```

The spectral-only signal is weak on `all_samples` after source-risk removal.

8. `full + text_koopman` lift:

Latest current `all_samples`:

```text
AUROC 0.6473
AUPRC 0.6263
TPR@FPR5% 0.0933
```

This matches the previous-best transition reference on TPR@FPR5% but does not
beat it on AUROC or AUPRC.

9. `transition + text_koopman` lift:

Latest current `all_samples`:

```text
AUROC 0.5221
AUPRC 0.5215
TPR@FPR5% 0.0733
```

This does not improve transition-state profiling.

10. `full + transition + text_koopman` vs previous best:

Latest current `all_samples`:

```text
AUROC 0.6632
AUPRC 0.6494
TPR@FPR5% 0.1267
```

Previous best transition reference:

```text
AUROC 0.6951
AUPRC 0.6592
TPR@FPR5% 0.0933
```

The new combination improves TPR@FPR5% but remains below the previous best on
AUROC and AUPRC.

11. Low-FPR:

Low-FPR improves for `full_plus_transition_plus_text_koopman` in the latest
current run:

```text
full_plus_text_koopman TPR@FPR5% = 0.0933
full_plus_transition_plus_text_koopman TPR@FPR5% = 0.1267
previous best TPR@FPR5% = 0.0933
```

12. Source/domain artifact vs pooled-z Deep DMD:

Architecturally, strict Text-Koopman is cleaner than pooled-z Deep DMD because
the classifier never receives pooled hidden states or pooled latent `z`.
Empirically, the full spectral set still encoded source artifacts, so the
pipeline required iterative source guarding to pass the configured source audit.

13. Whether to run Qwen2.5-7B:

Not recommended yet. The 1.5B strict pipeline does not beat the previous best on
AUROC/AUPRC, and source-safe spectral-only signal is weak.

14. If effect is weak:

Strict spectral-only Koopman is implemented, but on the current data it does not
exceed transition-state profiling on the main external AUROC/AUPRC criteria.
Transition-state profiling remains the stronger empirical baseline.

## Remaining Gaps

The full requested 1.5B grid is not exhausted:

```text
requested_1_5b_combinations=36
full_public_train_dev_checkpoints=6
capped_or_partial_checkpoints=1
missing_checkpoints=29
```

Coverage is recorded in:

```text
results_text_koopman/text_koopman_full_grid_coverage.csv
```

The practical blocker is cost/benefit rather than implementation. Given weak
source-safe spectral-only external performance, expanding to the remaining grid
or 7B is not currently justified unless the goal is exhaustive negative
evidence rather than model improvement.
