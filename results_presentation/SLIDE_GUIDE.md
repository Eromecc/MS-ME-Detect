# Slide Guide

Slide 1:
Title: Public benchmarks show strong performance, but all_samples exposes a gap
Figure: slide01_source_matrix_auroc_heatmap.png
Takeaway: Models perform well on public splits but drop sharply on all_samples.

Slide 2:
Title: all_samples is a shifted external set
Figure: slide03_mean_smd_by_test_set.png
Takeaway: all_samples has much larger mean SMD than public test splits.

Slide 3:
Title: Shift concentrates in probability and scale-response features
Figure: slide04_top_shifted_features_all_samples.png
Takeaway: The largest shifts occur in Qwen probability and scale-response features.

Slide 4:
Title: Ghostbuster artifacts reverse on all_samples
Figure: slide06_ghostbuster_probability_reversal.png
Takeaway: Ghostbuster assigns higher AI probability to human text than AI text on all_samples.

Slide 5:
Title: Scale-response provides useful signal
Figure: slide07_ablation_scale_response_contribution.png
Takeaway: Basic+Scale outperforms Basic+Probability, supporting the value of scale-response profiling.

Slide 6:
Title: M4 is the closest public source, but low-FPR detection remains weak
Figure: slide08_m4_targeted_variants.png
Takeaway: M4-only performs best among public-source variants, but detection remains imperfect.

Slide 7:
Title: Transition remains the selected method after Deep DMD control experiments
Figure: method_comparison_deep_dmd_transition.png
Takeaway: Deep DMD was implemented and evaluated, but the selected main model remains leave_out_ghostbuster + full_plus_1_5b_and_7b_transition.

Slide 8:
Title: Deep DMD is a rigorous negative / secondary experiment
Figure: deep_dmd_negative_result_summary.png
Takeaway: Best fusion uses alpha=1.00, so the apparent fusion gain mainly selects transition score rather than a stable Deep DMD improvement.


Slide 9:
Title: Deep DMD transfers on public sources but not enough for all_samples
Figure: cross_source_deep_dmd_auroc_heatmap.png
Takeaway: Deep DMD captures public benchmark dynamics but does not close the external target shift.

Slide 10:
Title: Transition remains the selected main method
Figure: cross_source_deep_dmd_vs_transition_delta_auroc_heatmap.png
Takeaway: Deep DMD is complementary, but transition-state profiling remains more robust under current validation.
