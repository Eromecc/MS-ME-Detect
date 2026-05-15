# Method, Features, and Result Summary

Generated for the GitHub-facing project snapshot.

## Final Selected Model

The selected main model is:

`leave_out_ghostbuster + full_plus_1_5b_and_7b_transition`

It is selected because it gives the best robust external `all_samples` ranking among validated non-leaky models, while keeping model selection and thresholds on internal public dev splits only.

| Metric | all_samples value |
|---|---:|
| AUROC | 0.6951 |
| AUPRC | 0.6592 |
| F1 | 0.6799 |
| TPR@FPR=1% | 0.0200 |
| TPR@FPR=5% | 0.0933 |
| ECE | 0.1488 |
| Brier | 0.2459 |

## Feature Groups Used

The full cleaned feature model uses content-abstracted numerical signals rather than raw token IDs or token strings.

| Feature family | What it measures | Role in final pipeline |
|---|---|---|
| Basic burstiness | sentence length variation, punctuation ratios, repetition, compression, Zipf-style deviation | Lightweight baseline and sanity-check signal. |
| Structure | surface structural ratios and simple document-shape signals | Small baseline contribution. |
| Qwen probability summaries | PPL/loss distribution summaries from Qwen2.5-1.5B, 7B, and 14B | Useful but not sufficient alone; external transfer is limited. |
| Scale-response profiling | how loss/PPL changes across Qwen scales; slopes, gaps, ratios, curvature/area | Strongest handcrafted improvement before transition features. |
| Transition-state profiling | token-level loss sequence mapped into abstract states; transition matrices, entropy, runs, bursts, spectral gap | Current main improvement. It captures loss-dynamics structure without raw token IDs. |
| DMD-lite / Koopman spectral profiling | closed-form per-text linear operator on hand-built loss observables | Provides complementary signal but does not beat transition. |
| Deep DMD encoder | learnable lifting `g_theta(x_t)` plus Koopman operator `K` with multi-step prediction loss | Implemented and evaluated as a controlled secondary experiment; not selected as main method. |

## Which Feature Helped Most?

Ablation on `combined_public` external `all_samples` shows that scale-response is the first major gain over basic/probability-only features, and full cleaned features are strongest among the static full feature sets.

| feature_set | n_features | auroc | auprc | f1 | tpr_at_fpr_5pct | ece | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Basic only | 36.0000 | 0.5407 | 0.5353 | 0.6196 | 0.0667 | 0.1542 | 0.2861 |
| Probability only | 36.0000 | 0.5364 | 0.5592 | 0.4906 | 0.0867 | 0.1765 | 0.2969 |
| Scale-response only | 204.0000 | 0.5594 | 0.5795 | 0.5655 | 0.1133 | 0.1157 | 0.2648 |
| Basic + probability | 72.0000 | 0.5525 | 0.5659 | 0.5342 | 0.0867 | 0.1507 | 0.2781 |
| Basic + scale-response | 240.0000 | 0.6252 | 0.6155 | 0.6048 | 0.0467 | 0.0765 | 0.2459 |
| Full cleaned | 276.0000 | 0.6334 | 0.6224 | 0.6034 | 0.0667 | 0.0623 | 0.2440 |

Key interpretation:

- `probability_only` is not enough by itself.
- `scale_response_only` and `basic_plus_scale_response` outperform probability-heavy alternatives.
- `full_cleaned` improves AUROC/AUPRC over the original full feature set after removing invalid/32B placeholder columns.
- Transition-state profiling gives the strongest final selected model when trained with `leave_out_ghostbuster` and dual 1.5B+7B transition features.

## Transition-State Profiling

Transition features are built from token-level loss caches. Tokens are not used directly. Each token loss trajectory is converted into abstract loss states using train-only bins. The model then uses transition matrices and dynamics summaries:

- 3/5/7-state transition matrices
- transition entropy
- self/up/down transition rates
- large-jump rate
- high-loss burst density
- low/high-loss run lengths
- spectral gap

The best transition result is dual-scale 1.5B+7B transition with `leave_out_ghostbuster` training.

## Deep DMD Result

Deep DMD was fully implemented with learnable lifting and a Koopman operator, then evaluated in a full 1.5B+7B sweep.

Full sweep status:

| Item | Value |
|---|---:|
| architecture configs | 432 |
| summary rows | 216 |
| manifest errors | 0 |

| model_version | train_source | feature_set | auroc | auprc | f1 | tpr_at_fpr_5pct | ece | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Selected transition model | leave_out_ghostbuster | full_plus_1_5b_and_7b_transition | 0.6951 | 0.6592 | 0.6799 | 0.0933 | 0.1488 | 0.2459 |
| Deep DMD score-only | m4 | deep_dmd_score_only | 0.5392 | 0.5265 | 0.6535 | 0.0333 | 0.2033 | 0.3044 |
| Deep DMD spectral-only | combined_strict | deep_dmd_spectral_features_only | 0.5653 | 0.5449 | 0.0727 | 0.0267 | 0.3942 | 0.4026 |
| Full + Deep DMD | m4 | full_plus_deep_dmd_spectral | 0.6652 | 0.6571 | 0.6667 | 0.1200 | 0.4148 | 0.4099 |
| Full + transition + Deep DMD | leave_out_ghostbuster | full_plus_transition_plus_deep_dmd | 0.6894 | 0.6520 | 0.6897 | 0.1133 | 0.1505 | 0.2489 |
| Best fusion | leave_out_ghostbuster | ensemble_fusion | 0.6974 | 0.6628 | 0.6833 | 0.0933 | 0.1451 | 0.2443 |

Important conclusion: best fusion used `alpha=1.00`, so it effectively selected the transition-side score. Deep DMD is therefore documented as implemented and evaluated, but not selected as the main method.

## Deep DMD Cross-Source Matrix

To avoid judging Deep DMD only on `all_samples`, a cross-source matrix was added. It reused the full sweep for `m4`, `combined_strict`, and `leave_out_ghostbuster`, and added targeted qwen25_1_5b runs for `ghostbuster`, `hc3_plus`, `leave_out_m4`, and `leave_out_hc3_plus`.

Average Deep DMD delta versus transition/reference:

| subset | mean_delta_auroc | mean_delta_auprc | mean_delta_tpr_at_fpr5 |
| --- | --- | --- | --- |
| public tests | 0.0036 | 0.0024 | 0.0041 |
| all_samples | 0.0588 | 0.0335 | 0.0029 |

Generalization gaps:

| train_source | method | same_source_auroc | mean_public_cross_source_auroc | all_samples_auroc | same_to_all_gap | public_cross_to_all_gap |
| --- | --- | --- | --- | --- | --- | --- |
| combined_strict | deep_dmd_best_available |  | 0.9795 | 0.6621 |  | 0.3174 |
| ghostbuster | deep_dmd_best_available | 0.9975 | 0.8895 | 0.5079 | 0.4896 | 0.3816 |
| hc3_plus | deep_dmd_best_available | 0.9659 | 0.9254 | 0.5932 | 0.3727 | 0.3322 |
| leave_out_ghostbuster | deep_dmd_best_available |  | 0.9557 | 0.6894 |  | 0.2662 |
| leave_out_hc3_plus | deep_dmd_best_available |  | 0.9768 | 0.6134 |  | 0.3634 |
| leave_out_m4 | deep_dmd_best_available |  | 0.9455 | 0.4843 |  | 0.4611 |
| m4 | deep_dmd_best_available | 0.9949 | 0.9264 | 0.6708 | 0.3242 | 0.2556 |

Conclusion: Deep DMD has real public-source transfer signal, but it does not clearly reduce the public-to-`all_samples` gap. The stricter statement is not "Deep DMD is useless"; it is "Deep DMD is complementary, but transition-state profiling remains the more robust selected method under current validation."

## Reproducibility Map

Primary code:

- `src/feature_probability.py`: probability summaries and optional token-loss cache writer.
- `src/feature_scale_response.py`: scale-response features.
- `src/feature_transition_profile.py`: transition-state features from token-loss sequences.
- `src/feature_koopman_dmd.py`: DMD-lite spectral features.
- `src/deep_dmd_dataset.py`, `src/deep_dmd_model.py`, `src/deep_dmd_train.py`, `src/deep_dmd_features.py`: Deep DMD implementation.

Primary scripts:

- `scripts/run_source_matrix_eval.py`
- `scripts/run_transition_fullscale_optimized.py`
- `scripts/run_transition_7b_targeted.py`
- `scripts/run_koopman_dmd_experiment.py`
- `scripts/run_deep_dmd_experiment.py`
- `scripts/run_deep_dmd_cross_source_matrix.py`
- `scripts/organize_project_artifacts.py`
- `scripts/make_presentation_figures.py`

Key curated outputs for readers:

- `docs/RESULTS_SUMMARY.md`
- `docs/TRANSITION_STATE_PROFILING_SUMMARY.md`
- `docs/METHOD_FEATURES_AND_RESULTS.md`
- `results_curated/tables/`
- `results_curated/figures/`
- `results_presentation/figures_clean/`

Large files intentionally excluded from Git:

- raw/private datasets under `data/`
- model weights and cache directories
- `features_token_loss/`
- large feature CSV directories
- `checkpoints*/`
- full experiment result directories outside curated summaries
