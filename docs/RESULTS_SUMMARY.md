# Results Summary

Generated: 2026-05-14T03:18:39.376722+00:00


## Basic Baseline

Purpose: Initial external baseline.

Source: `results_external/external_eval_summary_basic.csv`


| train_set       | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    |
| --------------- | ------ | ------ | ------ | --------------- | ------ |
| ghostbuster     | 0.5637 | 0.5690 | 0.2618 | 0.0867          | 0.1622 |
| combined_public | 0.5440 | 0.5326 | 0.6304 | 0.0533          | 0.1630 |
| hc3_plus        | 0.5382 | 0.5281 | 0.5195 | 0.0467          | 0.2557 |
| m4              | 0.4928 | 0.5082 | 0.6667 | 0.0000          | 0.4648 |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Full Allfeatures

Purpose: Original full feature set with probability and scale-response summaries.

Source: `results_external/external_eval_summary_full_allfeatures.csv`


| train_set       | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    |
| --------------- | ------ | ------ | ------ | --------------- | ------ |
| combined_public | 0.6199 | 0.6058 | 0.6093 | 0.0533          | 0.0462 |
| m4              | 0.6135 | 0.5951 | 0.6667 | 0.0400          | 0.4337 |
| hc3_plus        | 0.5877 | 0.5795 | 0.1600 | 0.0467          | 0.2452 |
| ghostbuster     | 0.2593 | 0.3695 | 0.0412 | 0.0067          | 0.4278 |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Cleaned / Tuned

Purpose: Cleaned full_allfeatures and tuned comparison.

Source: `results_optimized/combined_public_cleaned_tuned_comparison.csv`


| model_version                             | n_features | best_model    | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    | brier_score |
| ----------------------------------------- | ---------- | ------------- | ------ | ------ | ------ | --------------- | ------ | ----------- |
| best_ablation_full_cleaned                | 276        | random_forest | 0.6334 | 0.6224 | 0.6034 | 0.0667          |        | 0.2440      |
| tuned_full_allfeatures                    | 276        | random_forest | 0.6324 | 0.6178 | 0.6225 | 0.0400          | 0.0781 | 0.2424      |
| cleaned_full_allfeatures                  | 276        | random_forest | 0.6303 | 0.6197 | 0.6118 | 0.0533          | 0.0825 | 0.2436      |
| original_combined_public_full_allfeatures | 311        | random_forest | 0.6199 | 0.6058 | 0.6093 | 0.0533          |        | 0.2465      |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Optimized External Eval

Purpose: Optimized model external evaluation.

Source: `results_optimized/optimized_external_eval_summary.csv`


| model_version          | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    | brier_score |
| ---------------------- | ------ | ------ | ------ | --------------- | ------ | ----------- |
| tuned_full_allfeatures | 0.6324 | 0.6178 | 0.6225 | 0.0400          | 0.0781 | 0.2424      |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Source Generalization

Purpose: Cross-source train/test matrix.

Source: `results_source_matrix/source_generalization_matrix.csv`


| train_name                     | test_name               | best_model    | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    |
| ------------------------------ | ----------------------- | ------------- | ------ | ------ | ------ | --------------- | ------ |
| train_ghostbuster              | ghostbuster_strict_test | random_forest | 0.9977 | 0.9985 | 0.9805 | 0.9871          | 0.0473 |
| train_leave_out_m4             | ghostbuster_strict_test | random_forest | 0.9977 | 0.9984 | 0.9793 | 0.9922          | 0.0482 |
| train_m4                       | m4_strict_test          | random_forest | 0.9942 | 0.9990 | 0.9822 | 0.9760          | 0.0268 |
| train_leave_out_ghostbuster    | m4_strict_test          | random_forest | 0.9925 | 0.9986 | 0.9813 | 0.9664          | 0.0324 |
| train_combined_strict          | ghostbuster_strict_test | random_forest | 0.9922 | 0.9938 | 0.9632 | 0.9742          | 0.0584 |
| train_leave_out_hc3_plus       | m4_strict_test          | random_forest | 0.9919 | 0.9985 | 0.9816 | 0.9730          | 0.0328 |
| train_combined_strict          | m4_strict_test          | random_forest | 0.9911 | 0.9983 | 0.9827 | 0.9646          | 0.0348 |
| train_leave_out_hc3_plus       | ghostbuster_strict_test | random_forest | 0.9905 | 0.9924 | 0.9608 | 0.9561          | 0.0572 |
| train_balanced_combined_strict | ghostbuster_strict_test | random_forest | 0.9894 | 0.9926 | 0.9394 | 0.9483          | 0.0771 |
| train_balanced_combined_strict | hc3_plus_strict_test    | random_forest | 0.9666 | 0.9798 | 0.9216 | 0.8571          | 0.0662 |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## M4 Targeted

Purpose: M4-targeted training variants for all_samples.

Source: `results_targeted/m4_targeted_summary.csv`


| train_variant                       | best_model          | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    | brier_score |
| ----------------------------------- | ------------------- | ------ | ------ | ------ | --------------- | ------ | ----------- |
| m4_only_cleaned                     | random_forest       | 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 | 0.4146      |
| m4_plus_hc3                         | random_forest       | 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 | 0.2504      |
| m4_plus_hc3_without_extreme_domains | random_forest       | 0.6480 | 0.6225 | 0.6577 | 0.0600          | 0.1146 | 0.2493      |
| m4_generator_balanced               | random_forest       | 0.6177 | 0.6006 | 0.6697 | 0.0867          | 0.3399 | 0.3608      |
| m4_label_balanced                   | random_forest       | 0.6056 | 0.5972 | 0.6458 | 0.0667          | 0.2219 | 0.2911      |
| m4_plus_hc3_balanced                | random_forest       | 0.6047 | 0.5934 | 0.1287 | 0.0600          | 0.2606 | 0.3060      |
| m4_domain_balanced                  | logistic_regression | 0.5000 | 0.4759 | 0.5034 | 0.0000          | 0.4312 | 0.4514      |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## All Samples Model Comparison

Purpose: Comparison of prior and targeted all_samples models.

Source: `results_targeted/all_samples_model_comparison.csv`


| auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    | brier_score |
| ------ | ------ | ------ | --------------- | ------ | ----------- |
| 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 | 0.4146      |
| 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 | 0.4146      |
| 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 | 0.2504      |
| 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 | 0.2504      |
| 0.6480 | 0.6225 | 0.6577 | 0.0600          | 0.1146 | 0.2493      |
| 0.6334 | 0.6224 | 0.6034 | 0.0667          |        | 0.2440      |
| 0.6332 | 0.6187 | 0.6194 | 0.0467          | 0.0848 | 0.2442      |
| 0.6324 | 0.6178 | 0.6225 | 0.0400          | 0.0781 | 0.2424      |
| 0.6303 | 0.6197 | 0.6118 | 0.0533          | 0.0825 | 0.2436      |
| 0.6199 | 0.6058 | 0.6093 | 0.0533          |        | 0.2465      |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Transition 1.5B Optimized

Purpose: Full-scale 1.5B transition feature selection and model comparison.

Source: `results_transition/fullscale_1_5b_optimized/transition_optimized_summary.csv`


| train_name            | experiment              | test_set    | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    |
| --------------------- | ----------------------- | ----------- | ------ | ------ | ------ | --------------- | ------ |
| leave_out_ghostbuster | full_plus_transition    | all_samples | 0.6816 | 0.6480 | 0.6833 | 0.0733          | 0.1370 |
| combined_strict       | full_plus_transition    | all_samples | 0.6573 | 0.6251 | 0.6526 | 0.0533          | 0.0468 |
| m4                    | full_without_transition | all_samples | 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 |
| leave_out_ghostbuster | full_without_transition | all_samples | 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 |
| combined_strict       | full_without_transition | all_samples | 0.6332 | 0.6187 | 0.6194 | 0.0467          | 0.0848 |
| m4                    | full_plus_transition    | all_samples | 0.6181 | 0.6037 | 0.6667 | 0.0467          | 0.4275 |
| combined_strict       | transition_top100       | all_samples | 0.6063 | 0.5821 | 0.5333 | 0.0733          | 0.1150 |
| leave_out_ghostbuster | transition_top100       | all_samples | 0.6034 | 0.5811 | 0.6301 | 0.1133          | 0.1091 |
| combined_strict       | transition_summary_only | all_samples | 0.5995 | 0.5736 | 0.5333 | 0.0533          | 0.1116 |
| combined_strict       | transition_all          | all_samples | 0.5972 | 0.5751 | 0.6205 | 0.0867          | 0.1048 |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Transition 1.5B Late Fusion

Purpose: Late fusion of full and transition-only scores.

Source: `results_transition/fullscale_1_5b_optimized/late_fusion_summary.csv`


| train_name            | test_set    | alpha  | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    |
| --------------------- | ----------- | ------ | ------ | ------ | ------ | --------------- | ------ |
| m4                    | all_samples |        | 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 |
| m4                    | all_samples | 1.0000 | 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 |
| leave_out_ghostbuster | all_samples |        | 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 |
| leave_out_ghostbuster | all_samples | 1.0000 | 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 |
| combined_strict       | all_samples | 0.9000 | 0.6403 | 0.6244 | 0.6234 | 0.0467          | 0.0688 |
| combined_strict       | all_samples |        | 0.6332 | 0.6187 | 0.6194 | 0.0467          | 0.0848 |
| combined_strict       | all_samples |        | 0.5972 | 0.5751 | 0.6205 | 0.0867          | 0.1048 |
| leave_out_ghostbuster | all_samples |        | 0.5944 | 0.5780 | 0.6715 | 0.0867          | 0.1517 |
| m4                    | all_samples |        | 0.5109 | 0.5044 | 0.6182 | 0.0467          | 0.3987 |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Transition 1.5B Feature Selection

Purpose: Transition feature selection variants.

Source: `results_transition/fullscale_1_5b_optimized/feature_selection_summary.csv`


| train_name            | feature_set             |
| --------------------- | ----------------------- |
| m4                    | transition_all          |
| m4                    | transition_top20        |
| m4                    | transition_top50        |
| m4                    | transition_top100       |
| m4                    | transition_summary_only |
| m4                    | transition_l1_selected  |
| leave_out_ghostbuster | transition_all          |
| leave_out_ghostbuster | transition_top20        |
| leave_out_ghostbuster | transition_top50        |
| leave_out_ghostbuster | transition_top100       |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Transition 7B Targeted

Purpose: Targeted 7B transition validation against 1.5B and dual-scale features.

Source: `results_transition/qwen25_7b_targeted/transition_7b_summary.csv`


| train_name            | experiment                                     | test_set    | auroc  | auprc  | f1     | tpr_at_fpr_5pct | ece    | brier_score |
| --------------------- | ---------------------------------------------- | ----------- | ------ | ------ | ------ | --------------- | ------ | ----------- |
| leave_out_ghostbuster | full_plus_1_5b_and_7b_transition               | all_samples | 0.6951 | 0.6592 | 0.6799 | 0.0933          | 0.1488 | 0.2459      |
| leave_out_ghostbuster | full_plus_1_5b_transition                      | all_samples | 0.6816 | 0.6480 | 0.6833 | 0.0733          | 0.1370 | 0.2471      |
| leave_out_ghostbuster | full_plus_7b_transition                        | all_samples | 0.6683 | 0.6377 | 0.6766 | 0.1067          | 0.1475 | 0.2514      |
| combined_strict       | full_plus_1_5b_and_7b_transition               | all_samples | 0.6603 | 0.6252 | 0.6725 | 0.0800          | 0.0714 | 0.2386      |
| combined_strict       | full_plus_7b_transition                        | all_samples | 0.6585 | 0.6285 | 0.6667 | 0.0467          | 0.0531 | 0.2380      |
| combined_strict       | full_plus_1_5b_transition                      | all_samples | 0.6573 | 0.6251 | 0.6526 | 0.0533          | 0.0468 | 0.2373      |
| leave_out_ghostbuster | full_plus_transition_scale_response_1_5b_to_7b | all_samples | 0.6532 | 0.6297 | 0.6745 | 0.0667          | 0.1565 | 0.2598      |
| m4                    | full_without_transition                        | all_samples | 0.6514 | 0.6252 | 0.6667 | 0.0667          | 0.4160 | 0.4146      |
| leave_out_ghostbuster | full_without_transition                        | all_samples | 0.6486 | 0.6230 | 0.6543 | 0.0600          | 0.1251 | 0.2504      |
| m4                    | full_plus_7b_transition                        | all_samples | 0.6404 | 0.6609 | 0.6667 | 0.1133          | 0.4271 | 0.4259      |


Recommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.


## Transition 7B Probe Manifest


```json
{
  "label_in_sample_probe_accuracy": 0.8748547074777218,
  "source_in_sample_probe_accuracy": 0.7950406819062379,
  "domain_in_sample_probe_accuracy": 0.7047655947307245
}
```

## Deep DMD Full Sweep

Purpose: controlled full-scale evaluation of a learnable lifting function `g_theta` plus linear Koopman operator `K`, compared against transition-state profiling and DMD-lite spectral profiling.

Source files:

- `results_deep_dmd/full_sweep_1_5b_7b/deep_dmd_full_sweep_manifest.json`
- `results_deep_dmd/full_sweep_1_5b_7b/deep_dmd_vs_previous_best.csv`
- `results_deep_dmd/full_sweep_1_5b_7b/DEEP_DMD_FULL_SWEEP_REPORT.md`

Manifest status:

| Item | Value |
|---|---:|
| architecture configs | 432 |
| summary rows | 216 |
| manifest errors | 0 |
| device | cuda |

All `all_samples` results below are external-only. Model selection, threshold selection, fusion alpha selection, and early stopping used public internal dev splits only.

| Model version | Train source | Feature set | AUROC | AUPRC | F1 | TPR@FPR=1% | TPR@FPR=5% | ECE | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Previous best transition | leave_out_ghostbuster | full_plus_1_5b_and_7b_transition | 0.6951 | 0.6592 | 0.6799 | 0.0200 | 0.0933 | 0.1488 | 0.2459 |
| Best Deep DMD score-only | m4 | deep_dmd_score_only | 0.5392 | 0.5265 | 0.6535 | 0.0200 | 0.0333 | 0.2033 | 0.3044 |
| Best Deep DMD spectral-only | combined_strict | deep_dmd_spectral_features_only | 0.5653 | 0.5449 | 0.0727 | 0.0200 | 0.0267 | 0.3942 | 0.4026 |
| Best full + Deep DMD | m4 | full_plus_deep_dmd_spectral | 0.6652 | 0.6571 | 0.6667 | 0.0133 | 0.1200 | 0.4148 | 0.4099 |
| Best full + transition + Deep DMD | leave_out_ghostbuster | full_plus_transition_plus_deep_dmd | 0.6894 | 0.6520 | 0.6897 | 0.0133 | 0.1133 | 0.1505 | 0.2489 |
| Best fusion | leave_out_ghostbuster | ensemble_fusion | 0.6974 | 0.6628 | 0.6833 | 0.0200 | 0.0933 | 0.1451 | 0.2443 |

Interpretation:

- The full Deep DMD sweep was completed successfully, but Deep DMD is not selected as the main method.
- Best fusion has `best_model=alpha=1.00`, i.e. `alpha=1.00`. This means the fusion effectively selects the transition-side score rather than demonstrating stable additional Deep DMD gain.
- `full + transition + Deep DMD` improves F1 and TPR@FPR=5% in one comparison, but it does not beat the selected transition model on AUROC/AUPRC and is less stable as a main result.
- The final selected main model remains `leave_out_ghostbuster + full_plus_1_5b_and_7b_transition`.

Recommendation: present Deep DMD as a rigorous controlled secondary / negative experiment. It verifies that a learnable Koopman encoder was implemented and evaluated, but the simpler transition-state profiling remains the main method.
