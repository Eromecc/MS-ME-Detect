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

## Exclusion Rationale

These files are excluded because they are large, may contain raw data or model
intermediate caches, are reproducible from scripts, or are not appropriate for
ordinary GitHub storage.  The repository should contain enough code, commands,
curated tables, and clean figures for review without shipping raw datasets or
large model-derived caches.
