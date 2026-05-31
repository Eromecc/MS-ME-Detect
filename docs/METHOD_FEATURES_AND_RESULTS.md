# Method, Features, and Result Summary

This document is aligned with the current paper-release snapshot. Historical transition, Deep DMD, and Text-Koopman results are retained elsewhere for provenance, but they are not the final paper model.

## Final Paper Model

The current selected paper model is:

```text
MS-ME-Detect Qwen14 segment fusion
base evidence = token-level loss trajectory behavior + multi-scale response + embedding-pool/base evidence
segment branch = frozen Qwen2.5-14B head/tail segment probe
fusion = validation-selected score-level fusion
selection split = Fakespot-like validation only
external split = all_samples, evaluation only
```

External `all_samples` result:

| Metric | Value |
|---|---:|
| AUROC | 0.907822 |
| AUPRC | 0.926497 |
| TPR@FPR<=5% | 0.713333 |
| F1@0.5 | 0.838926 |
| Accuracy@0.5 | 0.840000 |
| Brier score | 0.131683 |
| ECE | 0.113796 |

The segment backbone is frozen and no LLM fine-tuning is performed. Candidate and alpha selection use only the Fakespot-like validation split. The external `all_samples` split is used only for final reporting.

## Feature Families

| Feature family | Role in current paper snapshot |
|---|---|
| token-level loss trajectory behavior | Final base evidence family |
| multi-scale response features | Final base evidence family |
| embedding-pool/base evidence | Final no-segment base branch |
| Qwen2.5-14B head/tail segment geometry | Final segment branch |
| surface style / probability summaries | Historical or framework-level supporting features; not the named final evidence family |
| transition-only, Deep DMD, Text-Koopman | Historical/exploratory provenance; not final paper model |

Detailed release schemas are in `paper_release/feature_schema/`.

## Historical Note

Earlier versions of this file described:

```text
leave_out_ghostbuster + full_plus_1_5b_and_7b_transition
AUROC 0.6951
```

as a selected main model. That was a previous selected model before the Qwen14 segment paper release. It is now historical.

Strict Text-Koopman and Deep DMD results remain useful as exploratory or diagnostic experiments, but they should not be mixed with the final paper result. The current paper final model is documented in `paper_release/README_paper_release.md`.

## Current Reproducibility Map

Release audit files:

- `paper_release/README_paper_release.md`
- `paper_release/configs/final_qwen14_fusion_config.yaml`
- `paper_release/tables/`
- `paper_release/predictions_anonymous/`
- `paper_release/feature_schema/`
- `paper_release/figure_data/`
- `paper_release/manifests/`

Primary current-method notes:

- `docs/FINAL_FUSION_MODEL_SPEC.md`
- `docs/FAKESPOT_REPRODUCTION_QWEN14_SEGMENT.md`
- `docs/HISTORICAL_EXPERIMENTS.md`

Primary scripts:

- `scripts/build_embedding_segment_features.py`
- `scripts/train_clean_probe_from_features.py`
- `scripts/incremental_blend_with_probe.py`
- `paper_release/scripts/recompute_external_metrics.py`
- `paper_release/scripts/make_paper_figures_from_source_data.py`

Large files intentionally excluded from Git:

- raw/private datasets under `data/`
- model weights and local model caches
- hidden-state caches
- token-loss caches
- embedding caches and full feature matrices
- checkpoints and local logs
