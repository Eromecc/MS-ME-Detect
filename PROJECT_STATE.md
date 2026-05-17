# PROJECT_STATE

## 0. GitHub Release Prep Status

Current goal: final GitHub upload preparation.  The repository should include
code, core scripts, docs, curated result tables, clean figures, and
reproducibility commands, while excluding raw data, model caches, hidden states,
token-loss caches, full features, checkpoints, and large intermediate files.

Release-prep actions completed in this pass:

- Updated `README.md` with final project conclusions, selected model metrics,
  strict mathematical Text-Koopman summary, directory guide, quick commands, and
  non-uploaded artifact policy.
- Added `docs/FINAL_EXPERIMENT_SUMMARY.md`.
- Added `docs/GITHUB_ARTIFACT_MANIFEST.md`.
- Updated `.gitignore` to exclude hidden states, token-loss caches, strict
  Text-Koopman intermediate outputs, checkpoints, model artifacts, `*.joblib`,
  `*.pt`, `*.npz`, and large generated feature outputs.
- Copied strict mathematical Text-Koopman small result summaries and plots into
  `results_curated/`.

Next release-prep actions:

1. Finish final safety checks (`git status`, `find . -type f -size +50M`,
   `git check-ignore`).
2. Stage only allowed paths.
3. Verify staged files contain no large/raw/cache/checkpoint artifacts.
4. Commit with:
   `Finalize MS-ME-Detect results and strict Koopman artifacts`
5. Try `git push origin main`; if blocked, report the manual push command.

## 1. Previous Strict-Math Goal

Implement a strict mathematical Text-Koopman experiment in `MS-ME-Detect`:

```text
Qwen hidden-state token trajectory H
  -> learned lifting g_theta: hidden_size -> observable_dim
  -> observable_dim > hidden_size
  -> per-document truncated exact-DMD local K_tilde_i
  -> spectral/residual scalar fingerprint
  -> classical classifier
```

The strict-math variant must not use PCA/random projection, pooled hidden,
pooled z, CLS, token ids, raw tokens, raw text, label-classification loss, or a
global shared Koopman parameter.

## 2. Completed Steps

- Added strict mathematical theory doc:
  - `docs/STRICT_MATHEMATICAL_TEXT_KOOPMAN.md`
- Added strict-math model:
  - `src/text_koopman_strict_math_model.py`
  - `StrictKoopmanLifting(hidden_size, observable_dim)`
  - Direct `hidden_states -> g_theta`, no projection.
- Added strict-math training:
  - `src/text_koopman_strict_math_train.py`
  - Reconstruction + local exact-DMD one-step + multistep + stability + variance loss.
  - No label loss, no all_samples, no classifier head, no global K.
- Added strict-math feature extraction:
  - `src/text_koopman_strict_math_features.py`
  - Per-document `K_tilde_i` via truncated exact DMD.
  - Emits only `strict_koopman_*` scalar spectral/residual/trajectory features.
- Added strict-math audit:
  - `src/text_koopman_strict_math_audit.py`
  - `docs/STRICT_TEXT_KOOPMAN_LEAKAGE_AUDIT.md`
  - `results_text_koopman_strict_math/leakage_audit.json`
  - `results_text_koopman_strict_math/strict_math_requirement_checklist.csv`
- Added strict-math runner:
  - `scripts/run_text_koopman_strict_math_experiment.py`
- Verified syntax:
  - `python -m py_compile src/text_koopman_strict_math_model.py src/text_koopman_strict_math_train.py src/text_koopman_strict_math_features.py src/text_koopman_strict_math_audit.py scripts/run_text_koopman_strict_math_experiment.py`
- Ran dry-run:
  - `python scripts/run_text_koopman_strict_math_experiment.py --dry_run --train_sources leave_out_ghostbuster --model qwen25_1_5b --max_rows_per_split 50 --seed 42`
- Ran small validation:
  - `python scripts/run_text_koopman_strict_math_experiment.py --train_sources leave_out_ghostbuster --model qwen25_1_5b --max_rows_per_split 300 --max_length 256 --observable_multiplier 2 --dmd_ranks 16 32 --epochs 5 --run_lifting_train --run_feature_extract --run_eval --run_audit --seed 42`
- Generated plots:
  - `python scripts/run_text_koopman_strict_math_experiment.py --train_sources leave_out_ghostbuster --model qwen25_1_5b --max_rows_per_split 300 --max_length 256 --observable_multiplier 2 --dmd_ranks 16 32 --epochs 5 --run_eval --run_audit --run_plots --seed 42`

## 3. Modified Files

New strict-math files:

- `docs/STRICT_MATHEMATICAL_TEXT_KOOPMAN.md`
- `docs/STRICT_TEXT_KOOPMAN_LEAKAGE_AUDIT.md`
- `scripts/run_text_koopman_strict_math_experiment.py`
- `src/text_koopman_strict_math_audit.py`
- `src/text_koopman_strict_math_features.py`
- `src/text_koopman_strict_math_model.py`
- `src/text_koopman_strict_math_train.py`
- `PROJECT_STATE.md`

New strict-math output roots:

- `features_text_koopman_strict_math/`
- `checkpoints_text_koopman_strict_math/`
- `results_text_koopman_strict_math/`

Existing prior Text-Koopman files from the previous task remain untracked and
were not reverted.

## 4. Key Commands And Output Paths

Dry-run output:

- `results_text_koopman_strict_math/strict_math_dry_run.json`

Small-validation checkpoints:

- `checkpoints_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_hidden_to_obs3072_rank16_len256_n300/`
- `checkpoints_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_hidden_to_obs3072_rank32_len256_n300/`

Small-validation features:

- `features_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_hidden_to_obs3072_rank16_len256_n300/`
- `features_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_hidden_to_obs3072_rank32_len256_n300/`

Results:

- `results_text_koopman_strict_math/strict_math_summary.csv`
- `results_text_koopman_strict_math/strict_math_all_samples_summary.csv`
- `results_text_koopman_strict_math/strict_math_vs_previous_best.csv`
- `results_text_koopman_strict_math/strict_math_source_matrix.csv`
- `results_text_koopman_strict_math/leakage_audit.json`
- `results_text_koopman_strict_math/strict_math_requirement_checklist.csv`
- `results_text_koopman_strict_math/STRICT_TEXT_KOOPMAN_REPORT.md`

Plots:

- `results_text_koopman_strict_math/plots/strict_math_vs_transition_barplot.png/pdf`
- `results_text_koopman_strict_math/plots/strict_math_low_fpr_comparison.png/pdf`
- `results_text_koopman_strict_math/plots/strict_math_source_matrix_heatmap.png/pdf`
- `results_text_koopman_strict_math/plots/eigenvalue_complex_plane_examples.png/pdf`
- `results_text_koopman_strict_math/plots/spectral_radius_by_label.png/pdf`
- `results_text_koopman_strict_math/plots/dmd_residual_by_label.png/pdf`

## 5. Failures / Warnings

- First small-validation attempt was stopped manually because hidden-cache
  coverage checking was scanning shards instead of using manifest ids.
  The runner was patched to prefer manifest ids.
- The first full small-validation completion failed only at report generation
  because `pandas.DataFrame.to_markdown()` requires optional dependency
  `tabulate`. The report writer was patched to use `to_string()` inside a text
  block. The command was rerun successfully.
- No OOM occurred in the successful small validation.
- No projection fallback was used.
- Full targeted strict-math run has not been launched yet.

## 6. Current Small-Validation Facts

From `results_text_koopman_strict_math/leakage_audit.json`:

- `passed=true`
- `hidden_size=1536`
- `observable_dim=3072`
- `uses_projection=false`
- `uses_global_K_parameter=false`
- `uses_label_classifier=false`
- `strict_math_failed_due_to_memory=false`
- `banned_feature_names=[]`
- `non_strict_feature_names=[]`

From `results_text_koopman_strict_math/strict_math_all_samples_summary.csv`:

```text
rank16 strict_koopman_spectral_only:
  AUROC 0.5203, AUPRC 0.5133, TPR@FPR5% 0.0600
rank16 full_plus_strict_koopman:
  AUROC 0.6184, AUPRC 0.6341, TPR@FPR5% 0.2000
rank16 transition_plus_strict_koopman:
  AUROC 0.5353, AUPRC 0.5333, TPR@FPR5% 0.0600
rank16 full_plus_transition_plus_strict_koopman:
  AUROC 0.6064, AUPRC 0.6153, TPR@FPR5% 0.1467

rank32 strict_koopman_spectral_only:
  AUROC 0.5401, AUPRC 0.5343, TPR@FPR5% 0.0733
rank32 full_plus_strict_koopman:
  AUROC 0.6115, AUPRC 0.6268, TPR@FPR5% 0.1800
rank32 transition_plus_strict_koopman:
  AUROC 0.5287, AUPRC 0.5291, TPR@FPR5% 0.0400
rank32 full_plus_transition_plus_strict_koopman:
  AUROC 0.6133, AUPRC 0.6157, TPR@FPR5% 0.1600
```

Previous best transition reference:

```text
AUROC 0.6951
AUPRC 0.6592
TPR@FPR5% 0.0933
```

Small validation improves low-FPR for some combined feature sets but does not
beat previous best transition on AUROC/AUPRC.

## 7. Next Minimal Executable Plan

1. Inspect `results_text_koopman_strict_math/STRICT_TEXT_KOOPMAN_REPORT.md`.
2. Decide whether to run the full targeted strict-math command:
   - `--train_sources leave_out_ghostbuster m4 combined_strict`
   - `--dmd_ranks 16 32 64`
   - `--epochs 30 --patience 5`
3. Only run full targeted if the cost is acceptable. The small validation proves
   strict-math feasibility but not superiority over transition-state profiling.
