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

