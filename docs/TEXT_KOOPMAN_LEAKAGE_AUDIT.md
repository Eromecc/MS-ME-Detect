> Historical note: this document predates or discusses experiments before the current Qwen14 segment fusion paper release. The current paper final model is `MS-ME-Detect Qwen14 segment fusion` with AUROC 0.907822 on `all_samples`, documented in `paper_release/README_paper_release.md`. Text-Koopman, Deep DMD, transition-only, and 0.6951/0.7120 rows in this file are historical/exploratory unless explicitly restated as the Qwen14 final model.

# Text-Koopman Leakage Audit

The strict Text-Koopman pipeline enforces a spectral-only downstream feature policy.

## Allowed Features

Downstream classifiers may use features with prefix:

```text
text_koopman_
```

These features are scalar summaries of per-document local DMD operators and their residual dynamics, including:

- eigenvalue magnitude summaries
- stability ratios
- eigenvalue angle entropy
- singular spectrum summaries
- DMD reconstruction error
- multi-step DMD residuals
- observable trajectory velocity/acceleration/norm summaries

## Forbidden Features

The downstream classifier must not use:

- pooled hidden state
- pooled latent `z`
- CLS embedding
- mean token embedding
- raw token id
- token string
- raw text

The experiment runner fails the audit if any selected downstream feature name contains:

```text
pooled
hidden_mean
hidden_std
z_mean
z_std
cls
token_id
token_text
embedding_mean
```

## Runtime Audit Output

The runner writes:

```text
results_text_koopman/leakage_audit.json
```

The JSON includes:

- `passed`
- `banned_feature_names`
- `policy`
- `banned_terms`
- `source_artifact_risk`
- `source_probe_risks`

If `passed=false`, the experiment should be treated as invalid.

If a source probe has higher dev accuracy than the label probe for an
experiment, `source_artifact_risk=true` and the affected experiments are listed
in `source_probe_risks`.

## Latest Runtime Result

Latest checked run:

```text
experiment=leave_out_ghostbuster_qwen25_1_5b_proj256_obs512_rank32_len256
leakage_audit.passed=true
banned_feature_names=[]
source_artifact_risk=false
label_probe_dev_accuracy=0.5921
source_probe_dev_accuracy=0.4429
domain_probe_dev_accuracy=0.3936
n_features_after_source_guard=2
```

Because `source_probe_dev_accuracy < label_probe_dev_accuracy`, the latest
source-guarded run no longer triggers the configured source-artifact risk
flag. The train/dev-only `--source_guard_mode iterative` filter removes spectral
features until the final multivariate source probe no longer exceeds the label
probe, without using `all_samples`.

This pass is conservative and should be interpreted as a leakage-aware
lower-bound setting, not as evidence that the full spectral feature set is
source/domain-invariant.
