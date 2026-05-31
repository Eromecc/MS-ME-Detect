> Historical note: this document predates or discusses experiments before the current Qwen14 segment fusion paper release. The current paper final model is `MS-ME-Detect Qwen14 segment fusion` with AUROC 0.907822 on `all_samples`, documented in `paper_release/README_paper_release.md`. Text-Koopman, Deep DMD, transition-only, and 0.6951/0.7120 rows in this file are historical/exploratory unless explicitly restated as the Qwen14 final model.

# Strict Text-Koopman Spectral-Only Pipeline

This document records why the strict Text-Koopman pipeline was added and how it differs from the earlier DMD-lite and Deep DMD experiments.

## Existing DMD-Lite

`src/feature_koopman_dmd.py` implements a practical DMD-lite spectral profiling module.

- Input: token-level `loss_sequence` from Qwen probability cache.
- Observable: handcrafted loss trajectory features such as raw loss, delta loss, rolling mean/std, per-document normalized loss, and loss-quantile states.
- Operator: one closed-form local DMD operator `K_i` is estimated per text.
- Output: per-document spectral, stability, residual, and trajectory summary features.
- Limitation: it does not use Qwen hidden-state trajectories and does not learn a lifting function.

In short, DMD-lite has per-text `K_i`, but it operates on handcrafted loss observables.

## Existing Deep DMD

`src/deep_dmd_model.py` implements a learnable Deep DMD-style encoder.

- Input: loss-derived token observables from `src/deep_dmd_dataset.py`.
- Lifting: `g_theta` maps each token observable into a latent state.
- Operator: `K` is a global shared model parameter.
- Classifier: the forward pass pools `z_t` and feeds the pooled latent vector to a classifier head.
- Limitation: pooled `z_t` can carry content, source, and domain signals. The downstream decision is therefore not spectral-only.

The existing Deep DMD experiment is best described as learned latent dynamics plus pooled latent classifier plus global-K regularization. It is not a per-document Koopman spectral fingerprint classifier.

## Strict Text-Koopman Pipeline

The new strict pipeline is implemented in:

- `src/hidden_state_cache.py`
- `src/text_koopman_model.py`
- `src/text_koopman_train.py`
- `src/text_koopman_features.py`
- `scripts/run_text_koopman_strict_experiment.py`

The intended mathematical path is:

```text
Qwen hidden-state trajectory H
  -> train-only projection
  -> learned lifting g_theta
  -> per-document local DMD operator K_i
  -> spectral/residual scalar fingerprint
  -> classical classifier
```

## Hidden-State Cache

The cache stores last-layer Qwen hidden states:

```text
features_hidden_states/{model_name}/{dataset_name}/shard_*.pt
features_hidden_states/{model_name}/{dataset_name}_hidden_state_manifest.json
```

Each record contains:

- `id`
- `text_hash`
- `seq_len`
- `hidden_states`
- `model_name`
- `max_length`
- `hidden_size`

It does not store raw text, raw token strings, or raw token ids. `hidden_size` is read from `model.config.hidden_size`; it is not hard-coded.

## Learned Lifting

`TextKoopmanLifting` maps projected hidden states into a higher-dimensional observable state.

Default configuration:

- train-only scaler and random Gaussian projector
- `projection_dim`: 128 or 256
- `observable_dim`: 512 or 1024
- requirement: `observable_dim > projection_dim`
- MLP + GELU + LayerNorm
- no label classifier
- no global shared `K`

The lifting network is trained without label loss. Early stopping uses dev reconstruction/local-DMD loss only. `all_samples` is never used for training, early stopping, calibration, threshold tuning, feature selection, or model selection.

## Local Per-Document Operator

For each text:

```text
Z = g_theta(projected_hidden)
X = Z[:-1]^T
Y = Z[1:]^T
K_i = local truncated DMD estimate
```

The implementation uses reduced-rank exact DMD for stability and speed. It extracts eigenvalue, singular spectrum, stability, residual, and trajectory summary features from each document-specific local operator.

## Downstream Classifier Policy

The downstream classifier may use only scalar spectral/residual/trajectory features whose names start with `text_koopman_`.

Forbidden downstream inputs:

- pooled hidden vectors
- pooled latent `z`
- CLS embedding
- mean token embedding
- raw token id
- token string
- raw text

`results_text_koopman/leakage_audit.json` and `docs/TEXT_KOOPMAN_LEAKAGE_AUDIT.md` record this policy.

## Evaluation

The runner supports:

- `text_koopman_spectral_only`
- `full_plus_text_koopman_spectral`
- `transition_plus_text_koopman_spectral`
- `full_plus_transition_plus_text_koopman_spectral`

The primary comparison is against the previous best transition model:

```text
leave_out_ghostbuster + full_plus_1_5b_and_7b_transition
AUROC 0.6951
AUPRC 0.6592
TPR@FPR=5% 0.0933
```

If strict Text-Koopman does not beat transition-state profiling, the correct conclusion is still useful: the theoretically cleaner spectral-only variant was implemented and evaluated, but transition-state profiling remains the stronger empirical method on the current data.
