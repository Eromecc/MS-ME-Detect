> Historical note: this document predates or discusses experiments before the current Qwen14 segment fusion paper release. The current paper final model is `MS-ME-Detect Qwen14 segment fusion` with AUROC 0.907822 on `all_samples`, documented in `paper_release/README_paper_release.md`. Text-Koopman, Deep DMD, transition-only, and 0.6951/0.7120 rows in this file are historical/exploratory unless explicitly restated as the Qwen14 final model.

# Strict Mathematical Text-Koopman Leakage Audit

Created at: 2026-05-17T16:09:17.122279+00:00

- audit_passed: `True`
- hidden_size: `1536`
- observable_dim: `3072`
- uses_projection: `False`
- uses_global_K_parameter: `False`
- uses_label_classifier: `False`
- strict_math_failed_due_to_memory: `False`
- banned_feature_names: `[]`
- non_strict_feature_names: `[]`

The downstream classifier is valid only when every selected feature has the `strict_koopman_` prefix and no banned pooled/token/text feature name appears in the selected feature list.
