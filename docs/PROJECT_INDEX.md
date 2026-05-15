# Project Index

Generated: 2026-05-14T03:18:39.373501+00:00

## Project Purpose

This project studies LLM-generated text detection. Current experiments focus on probability features, scale-response profiling, cross-source generalization diagnosis, and transition-state profiling from token-level loss sequences.

## Directory Guide

| Directory                  | Purpose                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| data/                      | Datasets, train/test splits, strict source splits; private data should not be committed. |
| data/raw/                  | Datasets, train/test splits, strict source splits; private data should not be committed. |
| data/train_sets/           | Datasets, train/test splits, strict source splits; private data should not be committed. |
| data/source_splits/        | Datasets, train/test splits, strict source splits; private data should not be committed. |
| features/                  | Generated feature CSVs for baseline feature extraction.                                  |
| features_by_dataset/       | Per-training-set feature tables, including full_allfeatures.                             |
| features_external/         | External all_samples feature tables.                                                     |
| features_source_matrix/    | Feature subsets extracted for strict source matrix experiments.                          |
| features_token_loss/       | Token-level loss cache for transition profiling; large generated cache.                  |
| features_transition/       | Transition-state profiling features generated from token-loss cache.                     |
| checkpoints/               | Original trained model checkpoints.                                                      |
| checkpoints_ablation/      | Ablation model checkpoints.                                                              |
| checkpoints_optimized/     | Cleaned/tuned model checkpoints.                                                         |
| checkpoints_source_matrix/ | Cross-source generalization model checkpoints.                                           |
| checkpoints_targeted/      | M4-targeted model checkpoints.                                                           |
| results/                   | Baseline result outputs.                                                                 |
| results_by_dataset/        | Per-dataset result outputs.                                                              |
| results_external/          | External evaluation results and feature audit summaries.                                 |
| results_optimized/         | Cleaned/tuned full_allfeatures results.                                                  |
| results_ablation/          | Feature ablation result tables and plots.                                                |
| results_source_matrix/     | Cross-source generalization matrices, shift reports, and plots.                          |
| results_diagnosis/         | all_samples diagnosis and error analysis plots/tables.                                   |
| results_targeted/          | M4-targeted training result tables and plots.                                            |
| results_transition/        | Transition-state profiling experiments and plots.                                        |
| results_presentation/      | Presentation-ready figures and slide guide.                                              |
| src/                       | Reusable project source code.                                                            |
| scripts/                   | Experiment and utility scripts.                                                          |
| logs/                      | Run logs and generated execution output.                                                 |

## What Should Be Committed To GitHub

Recommended:
- `src/`
- `scripts/`
- `docs/`
- README and small configuration files
- small CSV summaries that document headline results
- selected presentation figures from `results_curated/figures/`

Recommended not to commit:
- raw datasets and private test data
- local model files
- token-loss caches
- large feature CSVs
- checkpoint `.joblib` files
- large plot archives
- `__pycache__/` and `*.pyc`

## Current Best Result

Current best external `all_samples` setup:

`leave_out_ghostbuster + full_plus_1_5b_and_7b_transition`

| Metric | Value |
|---|---:|
| AUROC | 0.6951 |
| AUPRC | 0.6592 |
| F1 | 0.6799 |
| TPR@FPR=5% | 0.0933 |
| ECE | 0.1488 |
| Brier | 0.2459 |

## Main Conclusion

Public benchmark in-domain performance is very high, but `all_samples` is a strongly shifted external set. Ghostbuster-trained models show probability reversal on `all_samples`. Scale-response improves external ranking, and full-scale transition-state profiling further improves the best `all_samples` setup. Low-FPR detection remains weak and needs more calibration or training data alignment.
