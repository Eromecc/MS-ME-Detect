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

## Loss function update

The strict mathematical variant must not train `g_theta` as only an
autoencoder. Reconstruction alone can preserve hidden-state information without
forcing the lifted trajectory to be approximately linear under a local Koopman
operator. The training objective is therefore:

```text
L_total = L_recon + alpha * L_lin + beta * L_multi
```

where:

```text
L_recon = mse(H, Decoder(Z))
Z = g_theta(H)
X = Z[:-1]^T
Y = Z[1:]^T
U, S, Vh = svd(X, full_matrices=False)
K_tilde_i = U_r^T Y V_r diag(1 / (S_r + ridge))
L_lin = mse(K_tilde_i U_r^T X, U_r^T Y)
L_multi = mean_m mse(K_tilde_i^m U_r^T Z[:-m]^T, U_r^T Z[m:]^T)
```

The default weights are `alpha=1.0`, `beta=0.5`, and the default multi-step
horizons are `[2, 3]`. `K_tilde_i` is computed independently for each document
inside the forward loss path from the current lifted trajectory, so gradients
from `L_lin` and `L_multi` flow back through `Z` into `g_theta`.

The implementation remains strict:

- No global shared `K` parameter is introduced.
- No pooled hidden, pooled `z`, CLS, token id, token string, or raw text feature
  is used downstream.
- No label classification loss trains `g_theta`.
- Classical downstream classifiers receive only `strict_koopman_*` scalar
  spectral/residual/trajectory features computed from per-document local DMD.
- `all_samples` remains external test only and is not used for lifting
  training, early stopping, loss tuning, threshold selection, calibration, or
  model selection.

The runner supports `recon_only`, `recon_lin`, and `recon_lin_multi` loss modes
for ablation. `recon_only` is retained only as an ablation baseline.

## Loss-update result

The full `leave_out_ghostbuster` loss-update ablation completed with six
configurations: `recon_only`, `recon_lin`, and `recon_lin_multi` at DMD ranks
16 and 32. The manifest reported six successful configurations and zero errors.
The leakage audit passed.

Important caveat: the original strict mathematical training was not completely
reconstruction-only. It already contained DMD and multistep terms when DMD
succeeded, but the implementation was not sufficiently explicit for strict
ablation and could silently set dynamics terms to zero on DMD failure. The
updated implementation makes the loss mode explicit and records the individual
terms.

`all_samples` best overall:

```text
train_source = leave_out_ghostbuster
model = qwen25_1_5b
feature_set = full_plus_transition_plus_strict_koopman
loss_mode = recon_only
dmd_rank = 16
classifier = RandomForest
n_features = 805
AUROC 0.7120
AUPRC 0.6860
F1 0.6789
TPR@FPR1% 0.0133
TPR@FPR5% 0.1400
FPR@TPR95% 0.8400
ECE 0.1796
Brier 0.2569
MCC 0.1580
```

`all_samples` best low-FPR row:

```text
loss_mode = recon_lin
dmd_rank = 32
feature_set = full_plus_transition_plus_strict_koopman
AUROC 0.6577
AUPRC 0.6564
F1 0.6619
TPR@FPR5% 0.2067
```

Interpretation:

- The current best external result is the strict loss-update combined-feature
  row `recon_only/rank16`, not `recon_lin_multi`.
- `L_lin` helps some low-FPR settings, especially the `recon_lin/rank32`
  combined-feature row.
- `L_multi` did not show a further improvement in this full ablation.
- The strict mathematical invariants remain satisfied: no global shared `K`, no
  pooled-z classifier, no label loss for `g_theta`, and no `all_samples` use
  for training, model selection, thresholding, or calibration.
