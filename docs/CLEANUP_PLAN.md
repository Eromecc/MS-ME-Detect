# Cleanup Plan

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
