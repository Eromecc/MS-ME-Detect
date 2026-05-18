# GitHub Artifact Manifest

## Included Or Recommended For GitHub

- `src/`
- `scripts/`
- `docs/`
- `README.md`
- `PROJECT_STATE.md`
- `.gitignore`
- `results_curated/`
- `results_presentation/figures_clean/`
- `results_presentation/FIGURE_INDEX.md`
- `results_presentation/SLIDE_GUIDE.md`
- small files under `project_inventory/`
- `results_all_samples_scoreboard/` curated CSV/JSON/Markdown outputs only

## Excluded From GitHub

- `data/raw/`
- `data/dataset_english_v1.csv` when size/copyright status is unclear
- `features_token_loss/`
- `features_hidden_states/`
- `features_transition/formal/`
- `features_by_dataset/`
- `features_external/`
- `features_source_matrix/`
- `features_text_koopman/`
- `features_text_koopman_strict_math/`
- `checkpoints*/`
- `checkpoints_*/`
- `*.joblib`
- `*.jsonl.gz`
- `*.pt`
- `*.npz`
- large `all_features.csv` files
- model weights and local model caches

## all_samples External Scoreboard Artifacts

Included:

- `scripts/run_all_samples_external_scoreboard.py`
- `scripts/run_threshold_calibration_dev_only.py`
- `results_all_samples_scoreboard/model_family_comparison.csv`
- `results_all_samples_scoreboard/model_family_comparison.md`
- `results_all_samples_scoreboard/*/detector_metrics.csv`
- `results_all_samples_scoreboard/*/bootstrap_ci.csv`
- `results_all_samples_scoreboard/*/roc_curve.csv`
- `results_all_samples_scoreboard/*/pr_curve.csv`
- `results_all_samples_scoreboard/*/calibration_bins.csv`
- `results_all_samples_scoreboard/*/subgroup_metrics.csv`
- `results_all_samples_scoreboard/*/error_analysis.csv`
- `results_all_samples_scoreboard/*/*manifest.json`

Excluded:

- `data/test/all_samples_prepared.csv` and any raw `all_samples` text files
- `features_external/all_samples*/all_features.csv`
- `features_text_koopman_strict_math/**/all_samples_strict_koopman_features.csv`
- `checkpoints_text_koopman_strict_math/**/*.joblib`
- `checkpoints_text_koopman_strict_math/**/*.pt`
- token-loss caches, hidden-state caches, model downloads, and full feature matrices

Regeneration requires the private/generated artifacts above. The safe command
sequence is recorded in `docs/REPRODUCIBILITY_COMMANDS.md`. The scoreboard script
has a `--dry_run` readiness mode and a `--predictions_csv` curation mode that
does not load checkpoint pickles. Full checkpoint evaluation uses `joblib.load`
and should only be run in a trusted local environment because pickle/joblib
loading can execute code.

## Exclusion Rationale

These files are excluded because they are large, may contain raw data or model
intermediate caches, are reproducible from scripts, or are not appropriate for
ordinary GitHub storage.  The repository should contain enough code, commands,
curated tables, and clean figures for review without shipping raw datasets or
large model-derived caches.
