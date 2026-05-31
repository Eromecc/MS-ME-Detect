# MS-ME-Detect

MS-ME-Detect is a modular multi-evidence detector for LLM-generated text. The current paper-release model uses validation-selected score-level late fusion between a no-segment fused base score and a Qwen2.5-14B head/tail segment score.

## Current Paper-Release Model

Final score:

```text
s_final = clip((1 - 0.108) * rank01(s_base) + 0.108 * rank01(s_qwen14_segment), 0, 1)
```

The top-level final fusion uses two scores:

- no-segment fused base score
- Qwen14 segment score: `embedding_segment_qwen14_probe__sgd_a1e4`

Fusion details:

- transform: `rank01`
- alpha: `0.108`
- alpha/candidate selection: validation composite only
- threshold: fixed `0.5`
- external `all_samples`: final reporting only

The external `all_samples` benchmark is not used for training, candidate selection, alpha selection, threshold tuning, or calibration.

External metrics:

- AUROC: `0.907822`
- AUPRC: `0.926497`
- TPR@FPR<=5%: `0.713333`
- F1@0.5: `0.838926`
- Accuracy@0.5: `0.840000`
- MCC@0.5: `0.680060`
- Brier: `0.131683`
- ECE: `0.113796`

See:

- [paper_release/README_paper_release.md](paper_release/README_paper_release.md)
- [docs/FINAL_FUSION_MODEL_SPEC.md](docs/FINAL_FUSION_MODEL_SPEC.md)
- [docs/HISTORICAL_EXPERIMENTS.md](docs/HISTORICAL_EXPERIMENTS.md)

Historical Text-Koopman, Deep DMD, DMD-lite, and transition-only experiments are retained for provenance only and are not the final paper model.

## What Is Included

This repository includes source code, experiment scripts, small result tables, documentation, release manifests, feature schemas, figure source data, and anonymous hash-only prediction scores when available.

The paper-release directory is designed for review: files are small, text-readable, and auditable without downloading model caches or raw datasets.

## What Is Not Included

The repository intentionally excludes:

- raw datasets and raw text
- model weights
- hidden-state caches
- token-loss caches
- embedding caches and embedding matrices
- full feature matrices
- checkpoints
- local logs and wandb outputs

These files are large, may contain sensitive source data or model-derived caches, and are outside the GitHub review boundary.

## Reproducibility Boundary

Large artifacts must be regenerated locally from the documented scripts and manifests. The release snapshot provides small files for auditing reported metrics, feature families, selection boundaries, and figure source data.

To recompute external metrics from the anonymous release predictions:

```bash
python paper_release/scripts/recompute_external_metrics.py \
  --input paper_release/predictions_anonymous/all_samples_predictions_hash_only.csv
```

To create a lightweight audit sketch for Figure 3:

```bash
python paper_release/scripts/make_paper_figures_from_source_data.py
```

## Code Map

- [src/](src/): feature extraction, training, prediction, and exploratory dynamics modules
- [scripts/](scripts/): experiment orchestration, feature construction, evaluation, and figure helpers
- [docs/](docs/): method notes, historical experiment notes, and current final-model specs
- [paper_release/](paper_release/): manuscript release snapshot with small auditable files

## Use Caveat

MS-ME-Detect produces probabilistic risk scores, not proof of authorship. It should not be used as the sole basis for punishment or other high-stakes action. See [paper_release/ethics/INTENDED_USE_AND_LIMITATIONS.md](paper_release/ethics/INTENDED_USE_AND_LIMITATIONS.md).
