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

### Public-Safe Artifacts

These are suitable for a public-safe release because they are aggregate
summaries, reproducibility material, or code:

- `scripts/run_all_samples_external_scoreboard.py`
- `scripts/run_threshold_calibration_dev_only.py`
- `src/predict.py`
- `PROJECT_STATE.md`
- `docs/GITHUB_ARTIFACT_MANIFEST.md`
- `docs/REPRODUCIBILITY_COMMANDS.md`
- `results_all_samples_scoreboard/model_family_comparison.csv`
- `results_all_samples_scoreboard/model_family_comparison.md`
- `results_all_samples_scoreboard/*/detector_metrics.csv`
- `results_all_samples_scoreboard/*/bootstrap_ci.csv`
- `results_all_samples_scoreboard/*/roc_curve.csv`
- `results_all_samples_scoreboard/*/pr_curve.csv`
- `results_all_samples_scoreboard/*/calibration_bins.csv`
- `results_all_samples_scoreboard/*/subgroup_metrics.csv`
- `results_all_samples_scoreboard/*/*manifest.json`
- `results_text_koopman_strict_math/loss_update/*summary.csv`
- `results_text_koopman_strict_math/loss_update/*comparison.csv`
- `results_text_koopman_strict_math/loss_update/*REPORT.md`
- `results_text_koopman_strict_math/loss_update/*manifest.json`

### Internal-Only Uploaded Artifacts

These internal-only artifacts may contain sample IDs, labels, and model scores
derived from the external all_samples evaluation set. They are intended for
private/internal collaboration and should not be published in a public release.

- `results_all_samples_scoreboard/*/predictions.csv`
- `results_all_samples_scoreboard/*/error_analysis.csv`
- `results_all_samples_scoreboard/threshold_calibration_*/*.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/predictions.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/detector_metrics.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/metrics.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/roc_curve.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/pr_curve.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/calibration_bins.csv`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/classification_report.txt`
- `results_text_koopman_strict_math/loss_update/*_to_all_samples/*.png`

Excluded:

- `data/test/all_samples_prepared.csv` and any raw `all_samples` text files
- `features_external/all_samples*/all_features.csv`
- `features_text_koopman_strict_math/**/all_samples_strict_koopman_features.csv`
- `checkpoints_text_koopman_strict_math/**/*.joblib`
- `checkpoints_text_koopman_strict_math/**/*.pt`
- token-loss caches, hidden-state caches, model downloads, and full feature matrices
- `results_text_koopman_strict_math/loss_update/full_ablation_resume.log`
- `results_text_koopman_strict_math/loss_update/loss_update_manifest.partial.json`

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
## Fakespot-Like Segment Embedding Line

Public-safe additions for the current 0.907822 AUROC local reproduction line:

- `docs/FAKESPOT_REPRODUCTION_QWEN14_SEGMENT.md`
- `scripts/build_embedding_pool_features.py`
- `scripts/build_embedding_segment_features.py`
- `scripts/train_clean_probe_from_features.py`
- `scripts/incremental_blend_with_probe.py`
- `scripts/combine_feature_roots.py`
- `scripts/run_embedding_segment_multiseed_pipeline.sh`
- `scripts/run_embedding_segment_extra_followup.sh`
- `scripts/run_embedding_segment_llama_followup.sh`
- `scripts/run_embedding_segment_qwen7_followup.sh`

Internal-only artifacts for this line:

- `data/reproduction_datasets/*.csv`
- `data/test/all_samples_prepared.csv`
- `features_embedding_*` directories and `all_features.csv` matrices
- `results_embedding_segment_*` full prediction outputs
- `results_incremental_*` full prediction outputs
- downloaded model weights under local `models/`

Only aggregate metrics and method descriptions should be published unless the data owner explicitly approves publishing IDs, text, labels, and per-sample scores.

