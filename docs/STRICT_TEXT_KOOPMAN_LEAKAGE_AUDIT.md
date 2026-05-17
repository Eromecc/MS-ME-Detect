# Strict Mathematical Text-Koopman Leakage Audit

Created at: 2026-05-17T04:14:42.416245+00:00

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
