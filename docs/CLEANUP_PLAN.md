# Cleanup Plan

## Final GitHub Upload Cleanup

Keep and stage:

- `README.md`
- `PROJECT_STATE.md`
- `.gitignore`
- `src/`
- `scripts/`
- `docs/`
- `results_curated/`
- `results_presentation/figures_clean/`
- `results_presentation/FIGURE_INDEX.md`
- `results_presentation/SLIDE_GUIDE.md`
- small `project_inventory/*.{csv,json,txt}`

Do not stage:

- raw data
- large public/private dataset CSVs with unclear redistribution rights
- hidden-state shards
- token-loss caches
- transition formal feature caches
- full feature matrices
- checkpoints
- model weights
- `*.joblib`
- `*.pt`
- `*.npz`
- `*.jsonl.gz`

Before commit, run:

```bash
git status --short
find . -type f -size +50M | sort
git check-ignore -v data/raw features_token_loss features_hidden_states checkpoints 2>/dev/null || true
git diff --cached --stat
```

No files were deleted by this organization run.

## Cache Paths That Are Usually Safe To Delete After Listing

- `__pycache__/`
- `scripts/__pycache__/`
- `src/__pycache__/`

## Do Not Delete Without Archival

- `features_token_loss/`: expensive token-level cache
- `features_by_dataset/`: generated feature tables used by downstream scripts
- `features_external/`: external feature tables
- `checkpoints*/`: trained model checkpoints
- `results_transition/`: transition experiment records and plots

## Can Be Archived

- Older result directories after final summaries are copied to `results_curated/`
- Large plot archives not needed for presentation
- Intermediate predictions where summary tables already exist

## Do Not Move In Place

- `src/`, `scripts/`, `data/`, and generated feature directories are path dependencies for existing scripts.
