# Gitignore Suggestions

## Final Release Policy

The repository should commit code, docs, curated tables, and clean figures.  It
should not commit raw data, local model caches, hidden-state caches, token-loss
caches, full generated features, checkpoints, or model artifacts.

Required ignore classes now covered by `.gitignore`:

- `data/raw/`
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
- `*.parquet`
- `*.npz`
- `*.pt`
- `logs/`
- `results*/**/*.joblib`

Explicitly keep:

- `src/`
- `scripts/`
- `docs/`
- `results_curated/`
- `results_presentation/figures_clean/`
- `README.md`
- `PROJECT_STATE.md`

Do not overwrite `.gitignore` automatically. Suggested ignore rules:

- `__pycache__/`
- `*.pyc`
- `data/raw/`
- `models/`
- `features_token_loss/`
- `features_transition/formal/`
- `features_by_dataset/`
- `features_external/`
- `features_source_matrix/`
- `checkpoints*/`
- `results*/**/*.joblib`
- `logs/`
- `*.jsonl.gz`
- `*.parquet`
- `*.npz`
