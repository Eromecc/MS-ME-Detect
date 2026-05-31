# Historical Exploratory Experiments

This document separates historical MS-ME-Detect experiments from the current paper final model.

The current paper final model is documented in:

- `paper_release/README_paper_release.md`
- `paper_release/configs/final_qwen14_fusion_config.yaml`
- `docs/FINAL_FUSION_MODEL_SPEC.md`
- `docs/FAKESPOT_REPRODUCTION_QWEN14_SEGMENT.md`

## Current Paper Final Model

```text
MS-ME-Detect Qwen14 segment fusion
evidence = loss trajectory + scale response + embedding-pool/base evidence + Qwen2.5-14B head/tail segment branch
selection = Fakespot-like validation only
external benchmark = all_samples, evaluation only
AUROC = 0.907822
AUPRC = 0.926497
TPR@FPR<=5% = 0.713333
F1@0.5 = 0.838926
```

## Historical / Exploratory Results

The following experiment families are retained for provenance and method development. They are not the final paper model.

| Experiment family | Historical role | Not final because |
|---|---|---|
| transition-state profiling | Previous practical reference before embedding-pool and Qwen14 segment fusion | Superseded by current Qwen14 segment fusion paper snapshot |
| transition-only variants | Ablation and diagnostic variants | Not the selected final evidence fusion |
| Deep DMD | Controlled learnable-dynamics experiment | Did not provide stable external improvement over the selected validation policy |
| DMD-lite / Koopman spectral profiling | Exploratory dynamics feature family | Not selected as the final paper model |
| strict Text-Koopman | Theory-aligned hidden-state spectral experiment | Historical observed row; not the final paper release model |

Historical reference rows:

- Previous transition reference: `leave_out_ghostbuster + full_plus_1_5B_and_7B_transition`, AUROC 0.6951.
- Strict Text-Koopman observed row: `full_plus_transition_plus_strict_koopman / recon_only rank16`, AUROC 0.7120.

These rows may be cited as historical/exploratory provenance only. They should not be described as "current best", "final selected", or "paper final" in the current manuscript snapshot.

## Legacy Documents

Some older notes and drafts still contain detailed transition, Deep DMD, or Text-Koopman tables. Treat them as archival context unless they explicitly point to the current Qwen14 paper release.

Primary historical documents include:

- `docs/RESULTS_SUMMARY.md`
- `docs/TRANSITION_STATE_PROFILING_SUMMARY.md`
- `docs/TEXT_KOOPMAN_STRICT_PIPELINE.md`
- `docs/TEXT_KOOPMAN_COMPLETION_AUDIT.md`
- `docs/STRICT_MATHEMATICAL_TEXT_KOOPMAN.md`
- `docs/STRICT_TEXT_KOOPMAN_LOSS_UPDATE_AUDIT.md`
- `docs/PAPER_DRAFT_MS_ME_DETECT.md`
- `docs/PAPER_DRAFT_MS_ME_DETECT_v2_natural.md`

Keep these files for audit trail and method provenance, but use `paper_release/` for the current paper result.
