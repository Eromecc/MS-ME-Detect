# Transition-State Profiling Summary

## Motivation

Transition-state profiling tests a content/structure separation idea: instead of using raw token IDs or token strings, token-level losses are mapped into abstract states and summarized as transition patterns.

## Current Implementation

- token-level loss cache saved as compressed JSONL
- no raw token strings are stored
- loss quantile states
- 3/5/7-state transition matrices
- transition entropy, self-transition rate, upward/downward rates
- high-loss burst and low/high run-length features
- spectral gap
- train-only bins for state thresholds to avoid `all_samples` leakage

## 1.5B Fullscale Result

Best 1.5B setup:

`leave_out_ghostbuster + full_plus_transition`

| Metric | Value |
|---|---:|
| AUROC | 0.6816 |
| AUPRC | 0.6480 |
| F1 | 0.6833 |
| TPR@FPR=5% | 0.0733 |
| ECE | 0.1370 |

## 7B Targeted Result

Best targeted 7B setup:

`leave_out_ghostbuster + full_plus_1_5b_and_7b_transition`

| Metric | Value |
|---|---:|
| AUROC | 0.6951 |
| AUPRC | 0.6592 |
| F1 | 0.6799 |
| TPR@FPR=5% | 0.0933 |
| ECE | 0.1488 |
| Brier | 0.2459 |

## Interpretation

The transition signal is real: it improves the best external ranking setup and label probe accuracy remains higher than source/domain probes. Dual-scale 1.5B+7B transition features provide incremental value, while `leave_out_ghostbuster` remains the best training source for `all_samples`. Low-FPR detection remains weak, so feature selection, calibration, and source-aware validation should be prioritized before expensive 14B transition runs.

## Relation To Koopman-Inspired Idea

Transition-state profiling is Koopman-inspired but remains the selected practical method. A separate full Deep DMD encoder was later implemented and evaluated with a learnable lifting function `g_theta` and Koopman operator `K`; that controlled experiment did not provide stable additional external gain over transition-state profiling.


## Deep DMD Full Sweep Control

A full Deep DMD / Koopman encoder sweep was completed after the transition experiments.

| Item | Value |
|---|---:|
| Architecture configs | 432 |
| Summary rows | 216 |
| Manifest errors | 0 |

Best Deep DMD-related all_samples comparisons:

| Model | AUROC | AUPRC | F1 | TPR@FPR=5% | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Deep DMD score-only | 0.5392 | 0.5265 | 0.6535 | 0.0333 | 0.2033 | 0.3044 |
| Deep DMD spectral-only | 0.5653 | 0.5449 | 0.0727 | 0.0267 | 0.3942 | 0.4026 |
| Full + Deep DMD | 0.6652 | 0.6571 | 0.6667 | 0.1200 | 0.4148 | 0.4099 |
| Full + transition + Deep DMD | 0.6894 | 0.6520 | 0.6897 | 0.1133 | 0.1505 | 0.2489 |
| Best fusion | 0.6974 | 0.6628 | 0.6833 | 0.0933 | 0.1451 | 0.2443 |

The best fusion row used `alpha=1.00`, so it effectively selected the transition-side score. Deep DMD is therefore documented as an implemented but non-selected secondary experiment.

## Final Selected Main Model

`leave_out_ghostbuster + full_plus_1_5b_and_7b_transition` remains the final selected main model:

| Metric | Value |
|---|---:|
| AUROC | 0.6951 |
| AUPRC | 0.6592 |
| F1 | 0.6799 |
| TPR@FPR=1% | 0.0200 |
| TPR@FPR=5% | 0.0933 |
| ECE | 0.1488 |
| Brier | 0.2459 |
