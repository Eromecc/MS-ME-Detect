# Strict Mathematical Text-Koopman

This document defines the strict mathematical Text-Koopman experiment added as
a separate variant from earlier DMD-lite, Deep DMD, and projection-based strict
Text-Koopman runs.

## A. DMD-Lite

`src/feature_koopman_dmd.py` is a loss-sequence DMD baseline.

- input: token-level `loss_sequence`
- observable: handcrafted loss features
- operator: one per-text closed-form `K_i`
- limitation: no Qwen token hidden-state trajectory and no learned lifting

This baseline has local operators, but the state space is loss-derived and
handcrafted.

## B. Previous Deep DMD

`src/deep_dmd_model.py` is a learned latent dynamics model.

- input: loss-derived observable sequence
- lifting: learned encoder `g_theta`
- operator: global shared `K` parameter
- classifier: pooled `z_t` is passed to a classifier head
- limitation: not a per-document spectral-only classifier

The pooled latent classifier can carry content, source, and domain artifacts.
It is therefore not the desired local Koopman spectral fingerprint.

## C. Strict Mathematical Text-Koopman

The strict mathematical variant is implemented in:

- `src/text_koopman_strict_math_model.py`
- `src/text_koopman_strict_math_train.py`
- `src/text_koopman_strict_math_features.py`
- `src/text_koopman_strict_math_audit.py`
- `scripts/run_text_koopman_strict_math_experiment.py`

The path is:

```text
Qwen hidden-state token trajectory H
  -> learned lifting g_theta: R^hidden_size -> R^observable_dim
  -> observable trajectory Z
  -> per-document truncated exact-DMD K_tilde_i
  -> eigenvalue/singular-value/residual spectral fingerprint
  -> classical classifier
```

Strict requirements:

- `hidden_states` directly enter `g_theta`.
- No PCA projection.
- No random projection.
- No 128/256-dimensional bottleneck before lifting.
- `observable_dim > hidden_size`.
- For Qwen2.5-1.5B, `hidden_size` is read from cache/model config; observed
  cache manifests report `hidden_size=1536`, so the default strict observable
  dimension is `3072`.
- No label classifier in `g_theta`.
- No global shared `K`.
- `K_tilde_i` is estimated independently for each text.
- The downstream classifier receives only `strict_koopman_*` scalar spectral,
  residual, and observable-trajectory summary features.

Forbidden downstream inputs:

- pooled hidden state
- pooled latent `z`
- CLS embedding
- mean token embedding
- raw token id
- token string
- raw text

`all_samples` is external test only. It is not used for lifting training, early
stopping, threshold tuning, calibration, feature selection, or model selection.

The implementation uses truncated exact DMD:

```text
X = Z[:-1]^T
Y = Z[1:]^T
U, S, Vh = svd(X, full_matrices=False)
K_tilde = U_r^T Y V_r diag(1 / S_r)
```

This keeps the experiment mathematically aligned with local Koopman/DMD while
avoiding a dense full-observable inverse.
