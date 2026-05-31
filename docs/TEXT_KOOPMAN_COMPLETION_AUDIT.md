> Historical note: this document predates or discusses experiments before the current Qwen14 segment fusion paper release. The current paper final model is `MS-ME-Detect Qwen14 segment fusion` with AUROC 0.907822 on `all_samples`, documented in `paper_release/README_paper_release.md`. Text-Koopman, Deep DMD, transition-only, and 0.6951/0.7120 rows in this file are historical/exploratory unless explicitly restated as the Qwen14 final model.

# Text-Koopman Completion Audit

Audit timestamp: 2026-05-17 UTC.

## Objective

Implement and run a strict Text-Koopman spectral-only pipeline:

```text
Qwen hidden-state trajectory
  -> learned lifting g_theta
  -> per-document local K_i
  -> local-K spectral/residual scalar features
  -> classical classifier
```

The downstream classifier must not receive pooled hidden states, pooled latent
states, CLS/mean embeddings, token ids, token text, or raw text.

## Current Evidence

Implemented files:

- `docs/TEXT_KOOPMAN_STRICT_PIPELINE.md`
- `docs/TEXT_KOOPMAN_LEAKAGE_AUDIT.md`
- `src/hidden_state_cache.py`
- `src/text_koopman_model.py`
- `src/text_koopman_train.py`
- `src/text_koopman_features.py`
- `scripts/run_text_koopman_strict_experiment.py`

Output roots:

- `features_hidden_states/`
- `features_text_koopman/`
- `checkpoints_text_koopman/`
- `results_text_koopman/`

Verification commands run:

```bash
python -m py_compile src/hidden_state_cache.py src/text_koopman_features.py scripts/run_text_koopman_strict_experiment.py
python -m py_compile src/text_koopman_train.py
python -m py_compile scripts/run_text_koopman_strict_experiment.py src/hidden_state_cache.py src/text_koopman_model.py src/text_koopman_train.py src/text_koopman_features.py
python scripts/run_text_koopman_strict_experiment.py --dry_run --train_sources leave_out_ghostbuster --models qwen25_1_5b --max_rows_per_split 100 --seed 42
python scripts/run_text_koopman_strict_experiment.py --dry_run --train_sources leave_out_ghostbuster --models qwen25_1_5b --max_rows_per_split 100 --seed 42 --source_guard
python scripts/run_text_koopman_strict_experiment.py --train_sources leave_out_ghostbuster --models qwen25_1_5b --projection_dims 128 --observable_dims 512 --dmd_ranks 32 --max_length 256 --run_eval --source_guard --source_guard_max_source_acc 0.50 --source_guard_source_margin -0.05 --source_guard_min_features 12 --seed 42
python -m py_compile scripts/run_text_koopman_strict_experiment.py src/hidden_state_cache.py src/text_koopman_model.py src/text_koopman_train.py src/text_koopman_features.py
python scripts/run_text_koopman_strict_experiment.py --dry_run --train_sources leave_out_ghostbuster --models qwen25_1_5b --max_rows_per_split 100 --seed 42 --classifier_mode classical
python scripts/run_text_koopman_strict_experiment.py --train_sources leave_out_ghostbuster --models qwen25_1_5b --projection_dims 128 --observable_dims 512 --dmd_ranks 32 --max_length 256 --test_sets all_samples --run_eval --source_guard --source_guard_max_source_acc 0.50 --source_guard_source_margin -0.05 --source_guard_min_features 12 --classifier_mode classical --seed 42
python scripts/run_text_koopman_strict_experiment.py --train_sources leave_out_ghostbuster --models qwen25_1_5b --projection_dims 128 --observable_dims 512 --dmd_ranks 32 --max_length 256 --run_eval --source_guard --source_guard_max_source_acc 0.50 --source_guard_source_margin -0.05 --source_guard_min_features 12 --classifier_mode classical --seed 42
python - <<'PY'  # fake-model hidden cache OOM fallback smoke test; writes only to /tmp
...
PY
python - <<'PY'  # all_samples id overlap audit
...
PY
python - <<'PY'  # full targeted grid coverage audit
...
PY
python - <<'PY'  # PCA projector smoke test
...
PY
python scripts/run_text_koopman_strict_experiment.py --dry_run --train_sources leave_out_ghostbuster --models qwen25_1_5b --max_rows_per_split 100 --seed 42 --projector_type pca
python - <<'PY'  # random/PCA/linear projector forward smoke test
...
PY
python scripts/run_text_koopman_strict_experiment.py --dry_run --train_sources leave_out_ghostbuster --models qwen25_1_5b --max_rows_per_split 100 --seed 42 --projector_type linear
python - <<'PY'  # fake hidden-cache linear projector train/load/extract smoke test; writes only to /tmp
...
PY
python scripts/run_text_koopman_strict_experiment.py --train_sources leave_out_ghostbuster --models qwen25_1_5b --projection_dims 128 --observable_dims 512 --dmd_ranks 32 --max_length 256 --run_eval --source_guard --source_guard_mode iterative --source_guard_max_source_acc 0.50 --source_guard_source_margin 0.00 --source_guard_min_features 6 --classifier_mode classical --seed 42
python scripts/run_text_koopman_strict_experiment.py --train_sources leave_out_ghostbuster --models qwen25_1_5b --projection_dims 128 --observable_dims 512 --dmd_ranks 32 --max_length 256 --run_eval --source_guard --source_guard_mode iterative --source_guard_max_source_acc 0.50 --source_guard_source_margin 0.00 --source_guard_min_features 1 --classifier_mode classical --seed 42
python - <<'PY'  # machine-readable completion audit JSON
...
PY
python - <<'PY'  # backfill lifting training loss curve PNG/PDF from training_history.csv
...
PY
python - <<'PY'  # regenerate delta heatmap and eigen complex-plane proxy via runner plotting functions
...
PY
```

The `max_length=256` hidden-state cache now covers all required ids for:

- `m4_train/dev/test`
- `ghostbuster_train/dev/test`
- `hc3_plus_train/dev/test`
- `all_samples`

## Requirement Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Do not overwrite `data/dataset.csv` | No changes under tracked data files in `git status --short`. | Pass |
| Do not delete `data/dataset_english_v1.csv` | File was not touched. | Pass |
| Do not download new data | Pipeline uses local source splits and local model path checks. | Pass |
| Do not run 14B | Only `qwen25_1_5b` cache/checkpoints/results exist. | Pass |
| Do not rerun `scale_response` | No scale-response command was run; existing unrelated `features/scale_response_manifest.json` remains untracked from prior state. | Pass |
| Do not overwrite existing Deep DMD / DMD-lite / transition results | New outputs are under `features_hidden_states/`, `features_text_koopman/`, `checkpoints_text_koopman/`, `results_text_koopman/`. | Pass |
| Audit current DMD-lite and Deep DMD definitions | `docs/TEXT_KOOPMAN_STRICT_PIPELINE.md`. | Pass |
| Extract Qwen hidden-state cache | `features_hidden_states/qwen25_1_5b/*_hidden_state_manifest.json`; coverage verified for `max_length=256`. | Pass for 1.5B |
| Read `hidden_size` from `model.config.hidden_size` | `src/hidden_state_cache.py`; observed `hidden_size=1536` for Qwen2.5-1.5B. | Pass |
| Do not save raw token strings or raw token ids | Cache records store `id`, `text_hash`, `seq_len`, `hidden_states`, model metadata only. | Pass |
| GPU OOM fallback / smaller batch support | `src/hidden_state_cache.py` now recursively splits OOM batches and falls back to CPU for single-sample OOM; fake-model smoke test simulated OOM at batch size 2 and completed 3/3 rows with `failed_ids=[]`, `fallback_events=[split_batch]`, and `valid_len`. | Pass |
| `all_samples` external-only boundary | ID overlap audit: `all_samples_n=300`, overlap with public train/dev `0`, overlap with public test `0`; training code uses only `train_meta` for projector/lifting and dev unsupervised loss for early stopping. | Pass |
| Learned lifting with `observable_dim > projection_dim` | `TextKoopmanLifting` raises if not true. | Pass |
| Projector options | `HiddenProjector` supports train-only `random`, train-only `pca`, and `linear` where hidden_size -> projection_dim is trained inside `TextKoopmanLifting`; default runner option is `--projector_type random`. PCA and linear dry-runs verified; forward smoke test covered all three modes; fake hidden-cache linear train/load/extract smoke test produced 8 rows and 39 features with `failed=0`. | Pass |
| No label classifier inside lifting | `src/text_koopman_model.py` has encoder/decoder only. | Pass |
| Train lifting without labels/all_samples | Training loss in `src/text_koopman_train.py` is reconstruction/local-DMD/multistep/stability/variance; metadata states no labels/all_samples. | Pass in code |
| Per-document local K_i, no global shared K | `local_dmd_reduced` is called per record in training/features; model has no K parameter. | Pass |
| Downstream spectral-only features | `text_koopman_cols` selects `text_koopman_`; audit blocks banned names. | Pass |
| Feature CSV excludes pooled hidden/z/CLS/token/text | `src/text_koopman_features.py` emits scalar spectral/residual/trajectory summaries only. | Pass |
| Feature manifests are tied to checkpoint metadata | `checkpoint_created_at` is written in feature manifests; latest leave-out run has 10/10 manifests matching the current checkpoint. | Pass |
| Required metrics files per run | Existing result dirs contain `metrics.csv`, `predictions.csv`, `roc_curve.csv/png`, `pr_curve.csv/png`, `calibration_bins.csv/png`. | Pass for completed runs |
| Required `detector_metrics.csv` per run | `scripts/run_text_koopman_strict_experiment.py` now writes `detector_metrics.csv` after `eval_one`; 104 existing result dirs were backfilled from identical `metrics.csv`; audit found `detector_metrics_missing=0`. | Pass for completed runs |
| Summary/comparison/source matrix outputs | `results_text_koopman/text_koopman_summary.csv`, `text_koopman_all_samples_summary.csv`, `text_koopman_vs_previous_best.csv`, `text_koopman_source_matrix.csv`, `text_koopman_vs_transition_delta.csv`, plus `text_koopman_corrected_main_runs_summary.csv`. | Pass |
| Machine-readable completion audit | `results_text_koopman/text_koopman_completion_audit.json` records hard gates, latest all_samples metrics, probe rows, previous-best reference, and full-grid coverage. | Pass |
| Full targeted grid coverage matrix | `results_text_koopman/text_koopman_full_grid_coverage.csv`; audit found 36 requested 1.5B combinations, 6 full public train/dev checkpoints, 1 capped/partial checkpoint, and 29 missing. | Incomplete |
| Classical downstream candidates and dev model selection | `scripts/run_text_koopman_strict_experiment.py` now supports `--classifier_mode classical` by default: LogisticRegression, RandomForest, and XGBoost if importable; rank is dev AUPRC, AUROC, TPR@FPR5%, F1. Latest selection CSVs show LR/RF candidates; XGBoost was unavailable and skipped. | Pass |
| Leakage audit output | `results_text_koopman/leakage_audit.json`; banned feature audit passes. | Pass |
| Source/domain probe risk annotation | `source_artifact_risk` is written from probe comparison. Iterative train/dev-only source guard now has a leakage-safe setting with `source_artifact_risk=false` after retaining only one spectral feature. | Pass with conservative guard |
| Required plots | `results_text_koopman/plots/` contains all requested PNG/PDF plot stems, including delta heatmap and eigenvalue complex-plane examples. | Pass |
| Plot generation code path | `scripts/run_text_koopman_strict_experiment.py` now generates `text_koopman_delta_vs_transition_heatmap` and `eigenvalue_complex_plane_examples` in addition to the other requested plots; direct function call regenerated both PNG/PDF files. | Pass |
| Dry-run command | Ran successfully. | Pass |
| Small validation command | Ran successfully for leave-out ghostbuster, `proj128/256`, `obs512`, `rank16/32`. | Pass |
| Corrected full public train/dev run for `m4 + qwen25_1_5b + proj128 + obs512 + rank16` | Completed with `n_train_sequences_used=15623`, `n_dev_sequences_used=1967`, `best_epoch=15`, `early_stopping_epoch=20`; wrote metrics for all requested test sets. | Pass for this main combination |
| Corrected full public train/dev run for `combined_strict + qwen25_1_5b + proj128 + obs512 + rank16` | Completed with `n_train_sequences_used=21499`, `n_dev_sequences_used=2701`, `best_epoch=21`, `early_stopping_epoch=26`; wrote 16 metrics rows for all requested test sets. | Pass for this main combination |
| Corrected full public train/dev run for `leave_out_ghostbuster + qwen25_1_5b + proj128 + obs512 + rank16` | Completed with `n_train_sequences_used=16204`, `n_dev_sequences_used=2030`, `best_epoch=16`, `early_stopping_epoch=21`; wrote 16 metrics rows for all requested test sets. | Pass for this main combination |
| Corrected full public train/dev run for `leave_out_ghostbuster + qwen25_1_5b + proj256 + obs512 + rank16` | Completed with `n_train_sequences_used=16204`, `n_dev_sequences_used=2030`, `best_epoch=21`, `early_stopping_epoch=26`; all 10 feature manifests match checkpoint `created_at=2026-05-16T11:07:30.607666+00:00`; wrote 16 metrics rows for all requested test sets. | Pass for this main combination |
| Corrected full public train/dev run for `leave_out_ghostbuster + qwen25_1_5b + proj256 + obs512 + rank32` | Completed with `n_train_sequences_used=16204`, `n_dev_sequences_used=2030`, `best_epoch=21`, `early_stopping_epoch=26`, `best_dev_total=0.0117`; wrote the current 16 summary rows for all requested test sets. | Pass for this main combination |
| Full targeted command | The complete requested grid is not completed under corrected full public train/dev semantics. Remaining combinations are still missing or only have capped/small-validation checkpoints. | Missing |
| Qwen2.5-7B targeted | Not run, by design pending 1.5B gain. | Deferred |

## Important Validity Note

Older checkpoint metadata shows early runs used capped training:

```text
max_train_sequences=256
max_dev_sequences=64
effective_epochs_cap=5
```

The runner has been patched so future full runs reject those checkpoints as
incompatible and regenerate stale feature CSVs when checkpoint metadata changes.
One corrected full public train/dev combination has now completed:

```text
m4_qwen25_1_5b_proj128_obs512_rank16_len256
n_train_sequences_used=15623
n_dev_sequences_used=1967
best_epoch=15
early_stopping_epoch=20
```

One corrected full combined-strict public train/dev combination has also
completed:

```text
combined_strict_qwen25_1_5b_proj128_obs512_rank16_len256
n_train_sequences_used=21499
n_dev_sequences_used=2701
best_epoch=21
early_stopping_epoch=26
```

One corrected full leave-out-ghostbuster public train/dev combination has also
completed:

```text
leave_out_ghostbuster_qwen25_1_5b_proj128_obs512_rank16_len256
n_train_sequences_used=16204
n_dev_sequences_used=2030
best_epoch=16
early_stopping_epoch=21
```

The corrected full leave-out-ghostbuster `proj256` public train/dev combination
has also completed:

```text
leave_out_ghostbuster_qwen25_1_5b_proj256_obs512_rank16_len256
n_train_sequences_used=16204
n_dev_sequences_used=2030
best_epoch=21
early_stopping_epoch=26
checkpoint_created_at=2026-05-16T11:07:30.607666+00:00
feature_manifests_matching_checkpoint=10/10
```

The corrected full leave-out-ghostbuster `proj256/rank32` public train/dev
combination has also completed:

```text
leave_out_ghostbuster_qwen25_1_5b_proj256_obs512_rank32_len256
n_train_sequences_used=16204
n_dev_sequences_used=2030
best_epoch=21
early_stopping_epoch=26
best_dev_total=0.0117
checkpoint_created_at=2026-05-17T00:34:51.530807+00:00
```

The current run-specific `results_text_koopman/text_koopman_summary.csv`
contains the 16 evaluation rows for the corrected full `leave_out_ghostbuster`
`proj256/rank32` combination. A cross-run aggregate of the corrected main
combinations is written to:

```text
results_text_koopman/text_koopman_corrected_main_runs_summary.csv
```

This is still not the complete corrected full targeted grid across every
projection dimension, observable dimension, and DMD rank.

## Current Empirical Snapshot

Corrected full main-combination `all_samples` rows from
`results_text_koopman/text_koopman_corrected_main_runs_summary.csv`:

```text
m4 + text_koopman_spectral_only:
  AUROC 0.4436, AUPRC 0.4773, TPR@FPR5% 0.0400

m4 + full_plus_text_koopman:
  AUROC 0.5666, AUPRC 0.5509, TPR@FPR5% 0.0733

m4 + transition_plus_text_koopman:
  AUROC 0.4796, AUPRC 0.4866, TPR@FPR5% 0.0400

m4 + full_plus_transition_plus_text_koopman:
  AUROC 0.5400, AUPRC 0.5427, TPR@FPR5% 0.0000

combined_strict + text_koopman_spectral_only:
  AUROC 0.4216, AUPRC 0.4657, TPR@FPR5% 0.0333

combined_strict + full_plus_text_koopman:
  AUROC 0.6209, AUPRC 0.5823, TPR@FPR5% 0.0600

combined_strict + transition_plus_text_koopman:
  AUROC 0.4391, AUPRC 0.4762, TPR@FPR5% 0.0533

combined_strict + full_plus_transition_plus_text_koopman:
  AUROC 0.6167, AUPRC 0.5779, TPR@FPR5% 0.0467

leave_out_ghostbuster + text_koopman_spectral_only:
  AUROC 0.4562, AUPRC 0.4892, TPR@FPR5% 0.0533

leave_out_ghostbuster + full_plus_text_koopman:
  AUROC 0.6324, AUPRC 0.5937, TPR@FPR5% 0.0600

leave_out_ghostbuster + transition_plus_text_koopman:
  AUROC 0.5083, AUPRC 0.5106, TPR@FPR5% 0.0667

leave_out_ghostbuster + full_plus_transition_plus_text_koopman:
  AUROC 0.6205, AUPRC 0.5744, TPR@FPR5% 0.0133

leave_out_ghostbuster proj256/rank16 + text_koopman_spectral_only:
  AUROC 0.4801, AUPRC 0.4916, TPR@FPR5% 0.0267

leave_out_ghostbuster proj256/rank16 + full_plus_text_koopman:
  AUROC 0.6176, AUPRC 0.5780, TPR@FPR5% 0.0467

leave_out_ghostbuster proj256/rank16 + transition_plus_text_koopman:
  AUROC 0.5271, AUPRC 0.5194, TPR@FPR5% 0.0533

leave_out_ghostbuster proj256/rank16 + full_plus_transition_plus_text_koopman:
  AUROC 0.6014, AUPRC 0.5607, TPR@FPR5% 0.0067

leave_out_ghostbuster proj256/rank32 + text_koopman_spectral_only:
  AUROC 0.4892, AUPRC 0.4838, TPR@FPR5% 0.0000

leave_out_ghostbuster proj256/rank32 + full_plus_text_koopman:
  AUROC 0.6473, AUPRC 0.6263, TPR@FPR5% 0.0933

leave_out_ghostbuster proj256/rank32 + transition_plus_text_koopman:
  AUROC 0.5221, AUPRC 0.5215, TPR@FPR5% 0.0733

leave_out_ghostbuster proj256/rank32 + full_plus_transition_plus_text_koopman:
  AUROC 0.6632, AUPRC 0.6494, TPR@FPR5% 0.1267
```

Previous best transition reference:

```text
AUROC 0.6951, AUPRC 0.6592, TPR@FPR5% 0.0933
```

The strict Text-Koopman implementation currently does not provide enough
evidence to claim an overall improvement over transition-state profiling. The
latest corrected `proj256/rank32` all_samples run matches the previous-best
TPR@FPR5% for `full_plus_text_koopman` (0.0933) and improves it for
`full_plus_transition_plus_text_koopman` (0.1267 vs 0.0933), but AUROC/AUPRC
remain below the previous-best transition reference. The complete corrected
full targeted grid remains the main missing requirement.

Probe/audit for a corrected leave-out `proj256/rank16` run:

```text
leakage_audit.passed=true
banned_feature_names=[]
source_artifact_risk=false
experiment=leave_out_ghostbuster_qwen25_1_5b_proj256_obs512_rank16_len256
label_probe_dev_accuracy=0.7734
source_probe_dev_accuracy=0.7345
domain_probe_dev_accuracy=0.3970
```

After adding a train/dev-only `--source_guard`, the
`leave_out_ghostbuster_qwen25_1_5b_proj128_obs512_rank32_len256` evaluation
was rerun with multiple guard settings. A 6-feature setting still failed:

```text
leakage_audit.passed=true
banned_feature_names=[]
source_artifact_risk=true
label_probe_dev_accuracy=0.6631
source_probe_dev_accuracy=0.6911
domain_probe_dev_accuracy=0.3227
```

The extreme 1-feature setting passed the source-artifact audit:

```text
kept_feature=text_koopman_multistep_mse_k8
leakage_audit.passed=true
banned_feature_names=[]
source_artifact_risk=false
label_probe_dev_accuracy=0.5975
source_probe_dev_accuracy=0.4655
domain_probe_dev_accuracy=0.2493
```

This closes the configured leakage gate only under a conservative, low-signal
feature subset. It should not be interpreted as proof that the full spectral
feature set is source/domain-invariant.

Public split signal is strong for spectral-only features, for example:

```text
combined_strict -> m4_test spectral-only AUROC 0.8732, AUPRC 0.9728
combined_strict -> ghostbuster_test spectral-only AUROC 0.9430, AUPRC 0.9553
combined_strict -> hc3_plus_test spectral-only AUROC 0.8899, AUPRC 0.9246
leave_out_ghostbuster -> m4_test spectral-only AUROC 0.8799, AUPRC 0.9748
leave_out_ghostbuster -> ghostbuster_test spectral-only AUROC 0.9478, AUPRC 0.9593
leave_out_ghostbuster -> hc3_plus_test spectral-only AUROC 0.8849, AUPRC 0.9139
```

However, every corrected main-combination external `all_samples` result remains
below the transition baseline on AUROC/AUPRC, so the 1.5B strict spectral-only
result does not justify a Qwen2.5-7B targeted run yet.

## Current Confidence

I do not have 100% factual confidence that the current strict spectral-only
strategy achieves source/domain-invariant detection. I do have high confidence
that the implementation enforces the intended architectural constraint:
hidden-state trajectory -> learned lifting -> per-document local K_i ->
spectral/residual scalar classifier, with no pooled hidden/z/token/text inputs.

Remaining failure modes:

- Full spectral scalar summaries can still encode source/domain artifacts even
  when no raw content vector is passed downstream. The conservative iterative
  guard can pass the configured source audit but may retain very little signal.
- Current `all_samples` performance remains below the previous best transition
  baseline.
- The complete corrected full grid across all requested projection dimensions,
  observable dimensions, and DMD ranks has not been exhausted.
- Qwen2.5-7B is not justified until 1.5B shows a robust external gain.

Recommended fixes before further scale-up:

- Treat `source_artifact_risk=true` as a blocking validity warning for claims of
  source/domain separation.
- Add source-balanced or group-aware model selection across public dev sources,
  not just primary dev AUPRC.
- Evaluate source-adversarial or group-invariant lifting objectives if the goal
  is artifact suppression, while keeping the downstream classifier spectral-only.
- Prefer transition-state profiling as the empirical baseline unless strict
  Text-Koopman improves external low-FPR metrics under a passing source audit.

## Prompt-To-Artifact Checklist

This section maps the original explicit deliverables to concrete current
artifacts and notes remaining gaps.

| Deliverable / gate | Artifact evidence | Current state |
|---|---|---|
| `docs/TEXT_KOOPMAN_STRICT_PIPELINE.md` with DMD-lite, Deep DMD, and strict pipeline definitions | File exists and documents `feature_koopman_dmd.py`, `deep_dmd_model.py`, and hidden trajectory -> lifting -> local `K_i` -> spectral classifier. | Complete |
| Hidden-state cache module | `src/hidden_state_cache.py`; manifests under `features_hidden_states/qwen25_1_5b/`. | Complete for 1.5B |
| Hidden size not hard-coded | Code reads `model.config.hidden_size`; manifest/sample record shows `hidden_size=1536`. | Complete |
| No raw token string/id saved | Sample shard keys are `id`, `text_hash`, `seq_len`, `hidden_states`, `model_name`, `max_length`, `hidden_size`; no token/text keys except hash. | Complete |
| Learned lifting module | `src/text_koopman_model.py`; `TextKoopmanLifting` enforces `observable_dim > projection_dim`, MLP+GELU+LayerNorm, decoder only. | Complete |
| No label classifier in lifting | Model state and metadata set `uses_label_classifier=false`; no classifier head exists. | Complete |
| Lifting train with reconstruction/local-DMD/multistep/stability/variance and no all_samples | `src/text_koopman_train.py`; checkpoint metadata says `selection=dev unsupervised reconstruction/local-DMD loss; no labels and no all_samples`. | Complete |
| Train/dev loss curves | `src/text_koopman_train.py` now saves `training_loss_curves.png/pdf` alongside `training_history.csv`; existing 8 checkpoint dirs with history CSV were backfilled and `loss_curve_missing=0`. | Complete |
| Per-document local `K_i`, no global shared K | `src/text_koopman_features.py::local_dmd_reduced` is called per hidden record; checkpoint metadata has `uses_global_K_parameter=false`. | Complete |
| Spectral-only feature CSV | `features_text_koopman/.../*_text_koopman_features.csv`; sampled file has 40 columns, only `id` plus `text_koopman_*`, no banned names. | Complete |
| Downstream classifier feature audit | `results_text_koopman/leakage_audit.json` and `docs/TEXT_KOOPMAN_LEAKAGE_AUDIT.md`; latest extreme iterative guard passes the configured source-artifact audit after retaining one feature. | Complete, conservative guard required |
| Downstream classical classifier selection | `results_text_koopman/*_classifier_selection.csv`; latest all_samples validation used dev-selected RandomForest for `text_koopman_spectral_only`, `full_plus_text_koopman`, and `full_plus_transition_plus_text_koopman`, and LogisticRegression for `transition_plus_text_koopman`. | Complete |
| Required result files | 104 completed result dirs have `metrics.csv`, `detector_metrics.csv`, `predictions.csv`, ROC/PR/calibration CSV/PNG files. | Complete for completed dirs |
| Summary/comparison outputs | `text_koopman_summary.csv`, `text_koopman_all_samples_summary.csv`, `text_koopman_vs_previous_best.csv`, `text_koopman_source_matrix.csv`, `text_koopman_vs_transition_delta.csv`. | Complete |
| Machine-readable completion audit | `results_text_koopman/text_koopman_completion_audit.json`. | Complete |
| Probe output | `results_text_koopman/probe_summary.csv`; latest extreme iterative guard has label probe 0.5975 and source probe 0.4655. | Complete |
| Required plots | `results_text_koopman/plots/` contains requested PNG/PDF stems. | Complete |
| Dry-run command | Ran successfully with and without `--source_guard`. | Complete |
| Small validation / targeted validation | Multiple 1.5B validation/eval runs exist; hidden cache and selected checkpoints verified. Latest current all_samples validation (`leave_out_ghostbuster`, `proj256/rank32`): spectral-only AUROC 0.4892/AUPRC 0.4838; full+text AUROC 0.6473/AUPRC 0.6263/TPR@FPR5% 0.0933; transition+text AUROC 0.5221/AUPRC 0.5215/TPR@FPR5% 0.0733; full+transition+text AUROC 0.6632/AUPRC 0.6494/TPR@FPR5% 0.1267. | Partially complete |
| Full targeted grid across all requested `train_sources x projection_dims x observable_dims x dmd_ranks` | Not exhausted under corrected full public train/dev semantics. | Missing |
| Qwen2.5-7B targeted | Deferred because 1.5B strict Text-Koopman did not show robust external gain. | Intentionally not run |

Completion decision: implementation and validation artifacts are present for
the strict 1.5B pipeline, but the original full targeted grid remains
incomplete. The latest extreme iterative source-guard audit passes, but only by
retaining one spectral feature, so the full spectral feature set remains risky.
The goal should not be treated as fully complete if exhaustive full-grid
execution is a hard requirement.

Latest full-grid coverage audit:

```text
requested_1_5b_combinations=36
full_public_train_dev_checkpoints=6
capped_or_partial_checkpoints=1
missing_checkpoints=29
coverage_csv=results_text_koopman/text_koopman_full_grid_coverage.csv
```

Full public train/dev checkpoints currently present:

```text
m4_qwen25_1_5b_proj128_obs512_rank16_len256
leave_out_ghostbuster_qwen25_1_5b_proj128_obs512_rank16_len256
leave_out_ghostbuster_qwen25_1_5b_proj128_obs512_rank32_len256
leave_out_ghostbuster_qwen25_1_5b_proj256_obs512_rank16_len256
leave_out_ghostbuster_qwen25_1_5b_proj256_obs512_rank32_len256
combined_strict_qwen25_1_5b_proj128_obs512_rank16_len256
```

No Text-Koopman training/evaluation process is currently running.
