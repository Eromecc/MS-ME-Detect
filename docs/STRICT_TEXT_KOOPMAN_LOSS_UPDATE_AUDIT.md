# Strict Text-Koopman Loss Update Audit

Created at: 2026-05-17

## Checklist

- Original training was not purely reconstruction-only when DMD succeeded: it
  already had one-step DMD and fixed multi-step terms.
- The original implementation was incomplete for strict loss ablation because
  DMD failures silently zeroed the dynamics terms, multi-step horizons were
  hard-coded, and CLI loss modes were missing.
- `L_lin` has been implemented as a differentiable per-document reduced DMD
  prediction loss.
- `L_multi` has been implemented as a differentiable multi-step loss using the
  same per-document local `K_tilde_i`.
- Gradients from `L_lin` and `L_multi` flow through `K_tilde_i` and `Z` back to
  `g_theta`; the loss path does not use `detach()` or `torch.no_grad()`.
- The implementation still has no global shared `K`.
- The implementation still has no pooled-z classifier.
- The implementation still has no label classification loss for `g_theta`.
- Downstream strict classifiers remain spectral/residual/trajectory scalar
  feature classifiers over `strict_koopman_*` columns.
- `all_samples` is not used for training, early stopping, loss tuning, model
  selection, threshold selection, or calibration.

## Verification status

- Python syntax check passed for the edited training and runner files.
- A synthetic gradient check passed for `recon_lin_multi`: `L_lin` and
  `L_multi` were non-zero and `g_theta` parameters received non-zero gradients.
- Full loss ablation completed successfully:
  - manifest: `results_text_koopman_strict_math/loss_update/loss_update_manifest.json`
  - completed at: `2026-05-17T16:10:48.648046+00:00`
  - configurations: `6/6` completed
  - errors: `0`
  - `error_log.txt`: not present
- Leakage audit passed:
  - `passed=true`
  - `uses_global_K_parameter=false`
  - `uses_label_classifier=false`
  - `uses_projection=false`
  - `banned_feature_names=[]`
  - `non_strict_feature_names=[]`
- Generated outputs:
  - `loss_ablation_summary.csv`
  - `all_samples_summary.csv`
  - `final_loss_update_comparison.csv`
  - `bootstrap_ci_summary.csv`
  - `loss_curves.csv`
  - `loss_update_manifest.json`
  - `LOSS_UPDATE_REPORT.md`
  - plots under `results_text_koopman_strict_math/loss_update/plots/`

## Full ablation findings

On `all_samples`, strict spectral-only improved over `recon_only/rank16` when
adding `L_lin`, but `L_multi` did not further improve the strict spectral-only
metrics.

```text
strict_koopman_spectral_only:
  recon_only rank16:       AUROC 0.4706, AUPRC 0.4750, TPR@FPR5% 0.0200
  recon_only rank32:       AUROC 0.5422, AUPRC 0.5520, TPR@FPR5% 0.0800
  recon_lin rank16:        AUROC 0.5350, AUPRC 0.5728, TPR@FPR5% 0.1133
  recon_lin rank32:        AUROC 0.5453, AUPRC 0.5670, TPR@FPR5% 0.0800
  recon_lin_multi rank16:  AUROC 0.5226, AUPRC 0.5404, TPR@FPR5% 0.0733
  recon_lin_multi rank32:  AUROC 0.5274, AUPRC 0.5481, TPR@FPR5% 0.1000
```

For `full_plus_transition_plus_strict_koopman` on `all_samples`, the best
completed configuration was `recon_only/rank16`, not `recon_lin_multi`.

```text
full_plus_transition_plus_strict_koopman:
  recon_only rank16:       AUROC 0.7120, AUPRC 0.6860, TPR@FPR5% 0.1400
  recon_only rank32:       AUROC 0.6947, AUPRC 0.6702, TPR@FPR5% 0.1400
  recon_lin rank16:        AUROC 0.6856, AUPRC 0.6709, TPR@FPR5% 0.1133
  recon_lin rank32:        AUROC 0.6577, AUPRC 0.6564, TPR@FPR5% 0.2067
  recon_lin_multi rank16:  AUROC 0.6632, AUPRC 0.6498, TPR@FPR5% 0.1467
  recon_lin_multi rank32:  AUROC 0.6438, AUPRC 0.6333, TPR@FPR5% 0.0800
```

Against the previous best transition reference
(`AUROC 0.6951`, `AUPRC 0.6592`, `TPR@FPR5% 0.0933`), the best all-samples
combined strict result (`recon_only/rank16`) exceeded all three metrics, but the
new dynamics-loss variants did not improve AUROC/AUPRC over that ablation.
`recon_lin/rank32` gave the highest all-samples low-FPR recall among combined
feature sets (`TPR@FPR5% 0.2067`) while having lower AUROC/AUPRC.

No NaN, OOM, or DMD fallback instability was observed. The completed loss curves
recorded `train_dmd_fallbacks=0` and `dev_dmd_fallbacks=0` for all logged
epochs.

Bootstrap CI was generated for the two strict loss-update all-samples
prediction files. The previous-best transition prediction file was not found in
the expected location, so that row is marked `skipped_predictions_not_found` in
`bootstrap_ci_summary.csv`.

## Recommendation

Do not start a larger targeted run solely from this loss update. `L_lin` is
useful as a strict mathematical regularizer and improves strict spectral-only
over the weakest reconstruction-only baseline, but `recon_lin_multi` did not
clearly outperform `recon_only` in the full leave-out-Ghostbuster ablation.
Further targeted runs should only proceed after deciding whether the goal is
mathematical strictness, low-FPR recall, or best aggregate AUROC/AUPRC.
