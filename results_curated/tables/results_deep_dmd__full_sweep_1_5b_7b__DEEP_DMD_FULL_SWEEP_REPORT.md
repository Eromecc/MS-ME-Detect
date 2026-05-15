# Deep DMD Full Sweep Report

Completed full qwen25_1_5b + qwen25_7b Deep DMD sweep. all_samples was used only as external test; model selection and threshold selection used public internal dev only.

## Manifest

- Device: cuda
- Architecture configs: 432
- Summary rows: 216
- Errors: 0
- Output dir: `/vepfs-mlp2/queue010/20252203113/MS-ME-Detect/results_deep_dmd/full_sweep_1_5b_7b`

## Previous Best Comparison

| model_version | train_source | feature_set | auroc | auprc | f1 | tpr_at_fpr_1pct | tpr_at_fpr_5pct | fpr_at_tpr_95pct | ece | brier | mcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous_best_transition | leave_out_ghostbuster | full_plus_1_5b_and_7b_transition | 0.6951 | 0.6592 | 0.6799 | 0.0200 | 0.0933 |  | 0.1488 | 0.2459 |  |
| best_deep_dmd_score_only | m4 | deep_dmd_score_only | 0.5392 | 0.5265 | 0.6535 | 0.0200 | 0.0333 | 0.8784 | 0.2033 | 0.3044 | 0.0785 |
| best_deep_dmd_spectral | combined_strict | deep_dmd_spectral_features_only | 0.5653 | 0.5449 | 0.0727 | 0.0200 | 0.0267 | 0.9600 | 0.3942 | 0.4026 | -0.0459 |
| best_full_plus_deep_dmd | m4 | full_plus_deep_dmd_spectral | 0.6652 | 0.6571 | 0.6667 | 0.0133 | 0.1200 | 0.9333 | 0.4148 | 0.4099 | 0.0000 |
| best_full_plus_transition_plus_deep_dmd | leave_out_ghostbuster | full_plus_transition_plus_deep_dmd | 0.6894 | 0.6520 | 0.6897 | 0.0133 | 0.1133 | 0.8600 | 0.1505 | 0.2489 | 0.2261 |
| best_fusion | leave_out_ghostbuster | ensemble_fusion | 0.6974 | 0.6628 | 0.6833 | 0.0200 | 0.0933 | 0.8176 | 0.1451 | 0.2443 | 0.1963 |

## Best All-Samples Row By Feature Set

| train_name | model_name | feature_set | auroc | auprc | f1 | tpr_at_fpr_1pct | tpr_at_fpr_5pct | ECE | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m4 | qwen25_1_5b+qwen25_7b | deep_dmd_score_average_1_5b_7b | 0.5362 | 0.5480 | 0.6481 | 0.0267 | 0.0733 | 0.1083 | 0.2659 |
| m4 | qwen25_1_5b | deep_dmd_score_only | 0.5392 | 0.5265 | 0.6535 | 0.0200 | 0.0333 | 0.2033 | 0.3044 |
| m4 | qwen25_1_5b+qwen25_7b | deep_dmd_score_weighted_fusion_1_5b_7b | 0.5407 | 0.5305 | 0.6551 | 0.0267 | 0.0267 | 0.1832 | 0.2940 |
| m4 | qwen25_1_5b+qwen25_7b | deep_dmd_spectral_concat_1_5b_7b | 0.5360 | 0.5440 | 0.6574 | 0.0133 | 0.0533 | 0.2921 | 0.3440 |
| combined_strict | qwen25_1_5b | deep_dmd_spectral_features_only | 0.5653 | 0.5449 | 0.0727 | 0.0200 | 0.0267 | 0.3942 | 0.4026 |
| leave_out_ghostbuster | qwen25_1_5b | ensemble_fusion | 0.6974 | 0.6628 | 0.6833 | 0.0200 | 0.0933 | 0.1451 | 0.2443 |
| m4 | qwen25_1_5b+qwen25_7b | full_plus_deep_dmd_1_5b_7b | 0.6708 | 0.6485 | 0.6667 | 0.0200 | 0.0667 | 0.4187 | 0.4149 |
| leave_out_ghostbuster | qwen25_1_5b | full_plus_deep_dmd_score | 0.6596 | 0.6310 | 0.6756 | 0.0200 | 0.0533 | 0.1243 | 0.2473 |
| m4 | qwen25_7b | full_plus_deep_dmd_spectral | 0.6652 | 0.6571 | 0.6667 | 0.0133 | 0.1200 | 0.4148 | 0.4099 |
| leave_out_ghostbuster | qwen25_1_5b | full_plus_transition_plus_deep_dmd | 0.6894 | 0.6520 | 0.6897 | 0.0133 | 0.1133 | 0.1505 | 0.2489 |
| leave_out_ghostbuster | qwen25_1_5b+qwen25_7b | full_plus_transition_plus_deep_dmd_1_5b_7b | 0.6807 | 0.6501 | 0.6814 | 0.0267 | 0.0867 | 0.1516 | 0.2504 |
| leave_out_ghostbuster | qwen25_1_5b+qwen25_7b | transition_plus_deep_dmd_1_5b_7b | 0.5726 | 0.5624 | 0.4715 | 0.0200 | 0.0867 | 0.3402 | 0.3690 |

## Probe Summary

| train_name | model_name | experiment | label_probe_accuracy | source_probe_accuracy | domain_probe_accuracy |
| --- | --- | --- | --- | --- | --- |
| leave_out_ghostbuster | qwen25_1_5b | leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC | 0.8166 | 0.7490 | 0.3247 |
| leave_out_ghostbuster | qwen25_7b | leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | 0.7968 | 0.7759 | 0.3822 |
| m4 | qwen25_1_5b | m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC | 0.8194 |  | 0.3286 |
| m4 | qwen25_7b | m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | 0.7905 |  | 0.3789 |
| combined_strict | qwen25_1_5b | combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB | 0.8333 | 0.5370 | 0.3161 |
| combined_strict | qwen25_7b | combined_strict_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | 0.8235 | 0.5366 | 0.3080 |

## Low-FPR Transfer Summary

| train_name | model_name | experiment | feature_set | target_dev_fpr | threshold | dev_tpr | dev_fpr | all_samples_tpr | all_samples_fpr | external_fpr_blowup_ratio | all_samples_precision | all_samples_recall | all_samples_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_out_ghostbuster | qwen25_1_5b | leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC | ensemble_fusion | 0.0100 | 0.9133 | 0.8278 | 0.0072 | 0.0533 | 0.0203 | 2.7973 | 0.7273 | 0.0533 | 0.0994 |
| leave_out_ghostbuster | qwen25_1_5b | leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC | ensemble_fusion | 0.0500 | 0.7700 | 0.9441 | 0.0471 | 0.3067 | 0.1284 | 2.7256 | 0.7077 | 0.3067 | 0.4279 |
| leave_out_ghostbuster | qwen25_7b | leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0100 | 0.9133 | 0.8278 | 0.0072 | 0.0533 | 0.0203 | 2.7973 | 0.7273 | 0.0533 | 0.0994 |
| leave_out_ghostbuster | qwen25_7b | leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0500 | 0.7700 | 0.9441 | 0.0471 | 0.3067 | 0.1284 | 2.7256 | 0.7077 | 0.3067 | 0.4279 |
| m4 | qwen25_1_5b | m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC | ensemble_fusion | 0.0100 | 0.8943 | 0.8838 | 0.0079 | 0.8133 | 0.7095 | 90.1014 | 0.5374 | 0.8133 | 0.6472 |
| m4 | qwen25_1_5b | m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC | ensemble_fusion | 0.0500 | 0.7691 | 0.9562 | 0.0472 | 0.9600 | 0.9662 | 20.4516 | 0.5017 | 0.9600 | 0.6590 |
| m4 | qwen25_7b | m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0100 | 0.9026 | 0.8797 | 0.0079 | 0.6133 | 0.5000 | 63.5000 | 0.5542 | 0.6133 | 0.5823 |
| m4 | qwen25_7b | m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0500 | 0.7645 | 0.9586 | 0.0472 | 0.9467 | 0.9865 | 20.8806 | 0.4931 | 0.9467 | 0.6484 |
| combined_strict | qwen25_1_5b | combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0100 | 0.8667 | 0.8472 | 0.0092 | 0.0533 | 0.0270 | 2.9243 | 0.6667 | 0.0533 | 0.0988 |
| combined_strict | qwen25_1_5b | combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0500 | 0.6967 | 0.9458 | 0.0499 | 0.2267 | 0.1216 | 2.4369 | 0.6538 | 0.2267 | 0.3366 |
| combined_strict | qwen25_7b | combined_strict_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0100 | 0.8667 | 0.8472 | 0.0092 | 0.0533 | 0.0270 | 2.9243 | 0.6667 | 0.0533 | 0.0988 |
| combined_strict | qwen25_7b | combined_strict_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB | ensemble_fusion | 0.0500 | 0.6967 | 0.9458 | 0.0499 | 0.2267 | 0.1216 | 2.4369 | 0.6538 | 0.2267 | 0.3366 |

## Feature File Check

| file | rows_checked | numeric_cols | nan_cells | inf_cells |
| --- | --- | --- | --- | --- |
| combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB/hc3_plus_test_deep_dmd_features.csv | 266 | 30 | 0 | 0 |
| combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB/m4_test_deep_dmd_features.csv | 1965 | 30 | 0 | 0 |
| combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB/train_deep_dmd_features.csv | 5000 | 30 | 0 | 0 |
| combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB/dev_deep_dmd_features.csv | 2701 | 30 | 0 | 0 |
| combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB/all_samples_deep_dmd_features.csv | 298 | 30 | 0 | 0 |
| combined_strict_qwen25_1_5b_seq256_ld32_hd64_lr3em04_cfgB/ghostbuster_test_deep_dmd_features.csv | 670 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/hc3_plus_test_deep_dmd_features.csv | 266 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/m4_test_deep_dmd_features.csv | 1965 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/train_deep_dmd_features.csv | 5000 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/dev_deep_dmd_features.csv | 2030 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/all_samples_deep_dmd_features.csv | 298 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/ghostbuster_test_deep_dmd_features.csv | 670 | 30 | 0 | 0 |
| m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/hc3_plus_test_deep_dmd_features.csv | 266 | 30 | 0 | 0 |
| m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/m4_test_deep_dmd_features.csv | 1965 | 30 | 0 | 0 |
| m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/train_deep_dmd_features.csv | 5000 | 30 | 0 | 0 |
| m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/dev_deep_dmd_features.csv | 1967 | 30 | 0 | 0 |
| m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/all_samples_deep_dmd_features.csv | 298 | 30 | 0 | 0 |
| m4_qwen25_1_5b_seq256_ld8_hd64_lr1em03_cfgC/ghostbuster_test_deep_dmd_features.csv | 670 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/hc3_plus_test_deep_dmd_features.csv | 266 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/m4_test_deep_dmd_features.csv | 1965 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/train_deep_dmd_features.csv | 5000 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/dev_deep_dmd_features.csv | 2030 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/all_samples_deep_dmd_features.csv | 298 | 30 | 0 | 0 |
| leave_out_ghostbuster_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/ghostbuster_test_deep_dmd_features.csv | 670 | 30 | 0 | 0 |
| m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/hc3_plus_test_deep_dmd_features.csv | 266 | 30 | 0 | 0 |
| m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/m4_test_deep_dmd_features.csv | 1965 | 30 | 0 | 0 |
| m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/train_deep_dmd_features.csv | 5000 | 30 | 0 | 0 |
| m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/dev_deep_dmd_features.csv | 1967 | 30 | 0 | 0 |
| m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/all_samples_deep_dmd_features.csv | 298 | 30 | 0 | 0 |
| m4_qwen25_7b_seq256_ld32_hd64_lr3em04_cfgB/ghostbuster_test_deep_dmd_features.csv | 670 | 30 | 0 | 0 |

## Interpretation

- Deep DMD score-only and spectral-only have independent signal but weak external transfer.
- Best score fusion slightly exceeds previous best AUROC/AUPRC, but it is effectively alpha=1.00 in the saved row, so the gain comes from the existing transition-side score rather than a robust Deep DMD contribution.
- Full + transition + Deep DMD improves F1 and TPR@FPR=5% relative to previous best but has lower AUROC/AUPRC.
- Low-FPR threshold transfer remains unstable; all_samples FPR can inflate substantially under dev-selected thresholds.
