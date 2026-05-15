# Presentation Figure Index

## slide01_source_matrix_auroc_heatmap.png

- Status: generated
- Data source: `results_source_matrix/source_generalization_matrix.csv`
- Shows: AUROC source generalization heatmap.
- Takeaway: Models perform well on public splits but drop sharply on all_samples.
- Suggested PPT page: Slide 1

## slide02_source_matrix_auprc_heatmap.png

- Status: generated
- Data source: `results_source_matrix/source_generalization_matrix.csv`
- Shows: AUPRC source generalization heatmap.
- Takeaway: AUPRC confirms the all_samples external gap.
- Suggested PPT page: Slide 1

## slide03_mean_smd_by_test_set.png

- Status: generated
- Data source: `results_source_matrix/distribution_shift_summary.json`
- Shows: Mean SMD by test set.
- Takeaway: all_samples has much larger mean SMD than public test splits.
- Suggested PPT page: Slide 2

## slide04_top_shifted_features_all_samples.png

- Status: generated
- Data source: `results_source_matrix/distribution_shift_report.csv`
- Shows: Top shifted all_samples features.
- Takeaway: The largest shifts occur in Qwen probability and scale-response features.
- Suggested PPT page: Slide 3

## slide05_pca_feature_space_by_source.png

- Status: generated
- Data source: `existing cleaned full_allfeatures and strict test splits`
- Shows: PCA feature-space view by source.
- Takeaway: all_samples separates from public test splits.
- Suggested PPT page: Slide 2

## slide06_ghostbuster_probability_reversal.png

- Status: generated
- Data source: `results_source_matrix/train_ghostbuster_to_all_samples/predictions.csv`
- Shows: Ghostbuster probability distributions on all_samples.
- Takeaway: Ghostbuster assigns higher AI probability to human than AI text.
- Suggested PPT page: Slide 4

## slide07_ablation_scale_response_contribution.png

- Status: generated
- Data source: `results_ablation/combined_public_feature_ablation_summary.csv`
- Shows: Feature ablation AUROC/AUPRC.
- Takeaway: Basic+Scale outperforms Basic+Probability.
- Suggested PPT page: Slide 5

## slide08_m4_targeted_variants.png

- Status: generated
- Data source: `results_targeted/m4_targeted_summary.csv`
- Shows: M4-targeted variant performance.
- Takeaway: M4-only best matches all_samples but remains imperfect.
- Suggested PPT page: Slide 6

## slide09_all_samples_roc_comparison.png

- Status: generated
- Data source: `results_source_matrix/*_to_all_samples/predictions.csv`
- Shows: ROC curves on all_samples.
- Takeaway: M4-like models are strongest but still limited.
- Suggested PPT page: Slide 6

## slide10_all_samples_pr_comparison.png

- Status: generated
- Data source: `results_source_matrix/*_to_all_samples/predictions.csv`
- Shows: PR curves on all_samples.
- Takeaway: PR performance remains modest under shift.
- Suggested PPT page: Slide 6

## Deep DMD / Transition Method Comparison

Figure: `figures_clean/method_comparison_deep_dmd_transition.png`
Source: `results_curated/tables/method_comparison_deep_dmd_transition.csv` and Deep DMD / transition summary CSVs.
Takeaway: Transition-state profiling remains the selected main method; Deep DMD was evaluated but did not add stable independent gain.
Suggested slide: Method comparison and final method selection.

Figure: `figures_clean/deep_dmd_negative_result_summary.png`
Source: `results_deep_dmd/full_sweep_1_5b_7b/deep_dmd_vs_previous_best.csv` and manifest.
Takeaway: Deep DMD was implemented as learnable lifting plus Koopman dynamics, but it is a controlled secondary / negative result rather than the final method.
Suggested slide: Deep DMD control experiment.
## Deep DMD Cross-Source Matrix

Figure: `figures_clean/cross_source_deep_dmd_auroc_heatmap.png`
Source: `results_deep_dmd/cross_source_matrix/deep_dmd_best_available_matrix.csv`
Takeaway: Deep DMD is strong on many public source-to-source tests, but all_samples remains a shifted target.
Suggested slide: Deep DMD cross-source validation.

Figure: `figures_clean/cross_source_deep_dmd_vs_transition_delta_auroc_heatmap.png`
Source: `results_deep_dmd/cross_source_matrix/deep_dmd_vs_transition_delta.csv`
Takeaway: Deep DMD provides small public-transfer deltas, but does not change the selected main method.
Suggested slide: Deep DMD versus transition.
