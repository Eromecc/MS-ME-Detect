# Paper release snapshot for MS-ME-Detect

This directory is the paper-release snapshot for the current MS-ME-Detect manuscript. It is not a dump of every historical experiment in the repository.

The final paper model is:

```text
final model: MS-ME-Detect Qwen14 segment fusion
evidence families: token-level loss trajectory behavior, multi-scale response features, embedding-pool/base evidence, Qwen2.5-14B head/tail segment branch
segment backbone: Qwen2.5-14B, frozen, no LLM fine-tuning
selection split: Fakespot-like validation only
external benchmark: all_samples, evaluation only
external result: AUROC 0.907822, AUPRC 0.926497, TPR@FPR<=5% 0.713333, F1@0.5 0.838926
```

The final detector is a validation-selected score-level fusion of a no-segment evidence base and a Qwen2.5-14B head/tail segment probe. The segment backbone is frozen; no LLM weights are fine-tuned. The external `all_samples` set is used only for final reporting.

## Upload Boundary

This release intentionally excludes large or sensitive artifacts:

- raw text and raw datasets
- local model weights
- hidden-state caches
- token-loss caches
- embedding matrices
- checkpoints
- full feature matrices

These files are excluded because they are large, may contain raw text or model-derived caches, and are not appropriate for a GitHub paper snapshot. They should be regenerated locally from the documented scripts and manifests.

## Auditable Small Files

Reviewers can audit the result using only small files in this directory:

- `configs/`: final model configuration and upload boundary
- `tables/`: external metrics, baseline comparison, Qwen14 ablation summary, calibration summary
- `predictions_anonymous/`: hash-only external predictions, if generated from local raw prediction files
- `feature_schema/`: feature family and segment-branch schemas
- `figure_data/`: small source tables for paper figures
- `manifests/`: data, leakage, run, and environment manifests
- `scripts/`: lightweight metric recomputation and plotting scripts

Historical Text-Koopman, Deep DMD, and transition-only results are retained in the repository for provenance and method development, but they are exploratory or historical results. They are not the final paper model.

## How To Audit The Result

Run:

```bash
python paper_release/scripts/recompute_external_metrics.py --input paper_release/predictions_anonymous/all_samples_predictions_hash_only.csv
```

The script reads only the anonymous prediction table. It does not read raw text, model weights, hidden-state caches, token-loss caches, embedding caches, or full feature matrices.

If `all_samples_predictions_hash_only.csv` is a TODO template rather than a generated prediction table, regenerate it locally from the raw prediction files and do not commit raw text.
