# Deep DMD Cross-Source Generalization Report

Generated: 2026-05-15T02:12:22.222372+00:00

## Run Status

- Reused full sweep sources: combined_strict, leave_out_ghostbuster, m4
- Targeted trained sources: ghostbuster (checkpoint reused), hc3_plus (checkpoint reused), leave_out_m4 (checkpoint reused), leave_out_hc3_plus (checkpoint reused)
- Skipped/missing sources: none
- Errors: 0

## Answers

1. Deep DMD is strong on many same-source public tests when paired with full features, but score-only/spectral-only transfer is much weaker.
2. On overlapping public cross-source cells, mean AUROC delta versus transition/reference is 0.0036.
3. On all_samples, mean AUROC delta versus transition/reference is 0.0588; all_samples remains the hardest target.
4. Deep DMD does not clearly reduce the public-to-all_samples gap. The strongest all_samples row is still effectively tied to transition-side behavior.
5. Probe/source-artifact risk should still be treated as present; full-sweep probes showed source/domain information remains recoverable from Deep DMD features.
6. It is too strong to say Deep DMD is universally useless. The stricter conclusion is that under current features and validation, transition-state profiling is more robust and remains the selected main method.
7. If Deep DMD has value, it is complementary source-transfer signal, not a replacement for transition-state profiling.

## Best Deep DMD Matrix Rows

| train_source | test_set | feature_set | auroc | auprc | f1 | tpr_at_fpr_5pct | ece | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_strict | all_samples | full_plus_transition_plus_deep_dmd | 0.6621 | 0.6266 | 0.6647 | 0.0600 |  | 0.2381 |
| combined_strict | ghostbuster_test | full_plus_deep_dmd_score | 0.9920 | 0.9938 | 0.9656 | 0.9638 |  | 0.0380 |
| combined_strict | hc3_plus_test | full_plus_deep_dmd_spectral | 0.9541 | 0.9747 | 0.8478 | 0.8447 |  | 0.1356 |
| combined_strict | m4_test | full_plus_deep_dmd_score | 0.9925 | 0.9986 | 0.9857 | 0.9808 |  | 0.0235 |
| ghostbuster | all_samples | deep_dmd_score_only | 0.5079 | 0.5121 | 0.0261 | 0.0600 |  | 0.4859 |
| ghostbuster | ghostbuster_test | full_plus_deep_dmd_spectral | 0.9975 | 0.9984 | 0.9857 | 0.9922 |  | 0.0136 |
| ghostbuster | hc3_plus_test | deep_dmd_score_only | 0.9070 | 0.9467 | 0.8328 | 0.7081 |  | 0.1446 |
| ghostbuster | m4_test | deep_dmd_score_only | 0.8719 | 0.9748 | 0.8834 | 0.6043 |  | 0.1430 |
| hc3_plus | all_samples | full_plus_deep_dmd_spectral | 0.5932 | 0.5801 | 0.1078 | 0.0533 |  | 0.3013 |
| hc3_plus | ghostbuster_test | full_plus_deep_dmd_spectral | 0.9357 | 0.9578 | 0.8291 | 0.7674 |  | 0.1319 |
| hc3_plus | hc3_plus_test | full_plus_deep_dmd_spectral | 0.9659 | 0.9787 | 0.9125 | 0.8447 |  | 0.0741 |
| hc3_plus | m4_test | full_plus_deep_dmd_spectral | 0.9151 | 0.9817 | 0.8754 | 0.6145 |  | 0.1268 |
| leave_out_ghostbuster | all_samples | full_plus_transition_plus_deep_dmd | 0.6894 | 0.6520 | 0.6897 | 0.1133 |  | 0.2489 |
| leave_out_ghostbuster | ghostbuster_test | full_plus_transition_plus_deep_dmd | 0.9193 | 0.9452 | 0.8594 | 0.6925 |  | 0.1137 |
| leave_out_ghostbuster | hc3_plus_test | full_plus_deep_dmd_score | 0.9534 | 0.9749 | 0.8251 | 0.8447 |  | 0.1523 |
| leave_out_ghostbuster | m4_test | full_plus_deep_dmd_score | 0.9944 | 0.9990 | 0.9846 | 0.9766 |  | 0.0214 |
| leave_out_hc3_plus | all_samples | full_plus_deep_dmd_spectral | 0.6134 | 0.5946 | 0.6423 | 0.0267 |  | 0.2721 |
| leave_out_hc3_plus | ghostbuster_test | full_plus_deep_dmd_spectral | 0.9909 | 0.9930 | 0.9595 | 0.9535 |  | 0.0413 |
| leave_out_hc3_plus | hc3_plus_test | full_plus_deep_dmd_spectral | 0.9475 | 0.9691 | 0.7797 | 0.7888 |  | 0.1988 |
| leave_out_hc3_plus | m4_test | full_plus_deep_dmd_spectral | 0.9919 | 0.9985 | 0.9810 | 0.9784 |  | 0.0250 |
| leave_out_m4 | all_samples | deep_dmd_score_only | 0.4843 | 0.4918 | 0.0261 | 0.0467 |  | 0.4654 |
| leave_out_m4 | ghostbuster_test | full_plus_deep_dmd_spectral | 0.9975 | 0.9983 | 0.9767 | 0.9897 |  | 0.0242 |
| leave_out_m4 | hc3_plus_test | full_plus_deep_dmd_spectral | 0.9513 | 0.9711 | 0.9102 | 0.8261 |  | 0.0871 |
| leave_out_m4 | m4_test | full_plus_deep_dmd_spectral | 0.8876 | 0.9772 | 0.8722 | 0.6031 |  | 0.1391 |
| m4 | all_samples | full_plus_deep_dmd_1_5b_7b | 0.6708 | 0.6485 | 0.6667 | 0.0667 |  | 0.4149 |
| m4 | ghostbuster_test | full_plus_transition_plus_deep_dmd | 0.9030 | 0.9308 | 0.8472 | 0.5788 |  | 0.1247 |
| m4 | hc3_plus_test | full_plus_deep_dmd_score | 0.9498 | 0.9658 | 0.7594 | 0.6584 |  | 0.3132 |
| m4 | m4_test | full_plus_deep_dmd_score | 0.9949 | 0.9991 | 0.9839 | 0.9784 |  | 0.0206 |

## Generalization Gap

| train_source | method | same_source_auroc | mean_public_cross_source_auroc | all_samples_auroc | same_to_all_gap | public_cross_to_all_gap |
| --- | --- | --- | --- | --- | --- | --- |
| combined_strict | deep_dmd_best_available |  | 0.9795 | 0.6621 |  | 0.3174 |
| ghostbuster | deep_dmd_best_available | 0.9975 | 0.8895 | 0.5079 | 0.4896 | 0.3816 |
| hc3_plus | deep_dmd_best_available | 0.9659 | 0.9254 | 0.5932 | 0.3727 | 0.3322 |
| leave_out_ghostbuster | deep_dmd_best_available |  | 0.9557 | 0.6894 |  | 0.2662 |
| leave_out_hc3_plus | deep_dmd_best_available |  | 0.9768 | 0.6134 |  | 0.3634 |
| leave_out_m4 | deep_dmd_best_available |  | 0.9455 | 0.4843 |  | 0.4611 |
| m4 | deep_dmd_best_available | 0.9949 | 0.9264 | 0.6708 | 0.3242 | 0.2556 |

## Probe Summary

| train_source | model_name | label_probe_accuracy | source_probe_accuracy | domain_probe_accuracy |
| --- | --- | --- | --- | --- |
| leave_out_ghostbuster | qwen25_1_5b | 0.8166 | 0.7490 | 0.3247 |
| leave_out_ghostbuster | qwen25_7b | 0.7968 | 0.7759 | 0.3822 |
| m4 | qwen25_1_5b | 0.8194 |  | 0.3286 |
| m4 | qwen25_7b | 0.7905 |  | 0.3789 |
| combined_strict | qwen25_1_5b | 0.8333 | 0.5370 | 0.3161 |
| combined_strict | qwen25_7b | 0.8235 | 0.5366 | 0.3080 |
| ghostbuster | qwen25_1_5b | 0.8738 |  | 0.6585 |
| hc3_plus | qwen25_1_5b | 0.7900 |  |  |
| leave_out_hc3_plus | qwen25_1_5b | 0.8252 | 0.6584 | 0.2595 |
| leave_out_m4 | qwen25_1_5b | 0.8647 | 0.7088 | 0.5027 |

Probe interpretation: label is generally recoverable from Deep DMD latent features, but source/domain information is also recoverable in multi-source settings. This supports treating source artifact risk as still present rather than fully solved.
