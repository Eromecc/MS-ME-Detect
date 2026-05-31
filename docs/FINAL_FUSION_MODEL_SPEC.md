# Final Fusion Model Specification

This note specifies the current strict final Fakespot-like external model used for the reported `all_samples` result.

## Executive Summary

The final model is a deterministic score-level fusion, not a newly trained high-dimensional classifier over all raw feature columns.

Final reported model:

```text
final artifact = artifacts/results/results_incremental_embedding_best_plus_embedding_segment_qwen14_v1/val_composite__all_samples_predictions.csv
base artifact  = artifacts/results/results_incremental_current_best_plus_embedding_pool_raw_plateau_v1/val_low_fpr__all_samples_predictions.csv
segment branch = Qwen2.5-14B head/tail segment probe
selected row   = embedding_segment_qwen14_probe__sgd_a1e4__rank
alpha          = 0.108
selection      = fakespot_like validation composite
report split   = external all_samples, evaluation only
threshold      = fixed 0.5 for thresholded metrics
```

Top-level fusion formula:

```text
s_final = clip((1 - alpha) * rank01(s_base) + alpha * rank01(s_qwen14_segment), 0, 1)
alpha = 0.108
```

where `rank01(x)` converts scores within the split to percentile ranks in `[0, 1]`. The selected validation objective is:

```text
validation composite = 0.45 * AUROC + 0.35 * AUPRC + 0.20 * TPR@FPR<=5%
```

The external `all_samples` labels are not used for candidate, alpha, threshold, or hyperparameter selection.

## Data Splits

| split | path | n | human | AI |
|---|---:|---:|---:|---:|
| train | `data/reproduction_datasets/fakespot_like_train.csv` | 132614 | 43920 | 88694 |
| validation/dev | `data/reproduction_datasets/fakespot_like_val.csv` | 14932 | 5031 | 9901 |
| external | `data/test/all_samples_prepared.csv` | 300 | 150 | 150 |

## Input Dimensionality

There are two relevant notions of dimensionality.

| level | input to model/fusion | dimensionality |
|---|---|---:|
| final top-level fusion | base score + Qwen14 segment candidate score, both rank-normalized | 2 score inputs |
| Qwen14 segment candidate probe | cached Qwen14 head/tail segment numeric features | 776 features |
| embedding-pool branch probe used inside the no-segment base | DeBERTa/RoBERTa pooled representation features | 518 features |
| earlier external reproduction feature matrix | deterministic/probability/scale/transition features | 720 features |
| cross-dynamics with72 feature matrix used in earlier ablations | cross-model loss disagreement + loss dynamics | 475 features |

The final deployed score is therefore a 2-input score fusion, but those two scores are produced by lower-level branches trained on the feature matrices above.

## Evidence Families and Feature Counts

| evidence family | cached feature root | feature count | notes |
|---|---|---:|---|
| Qwen14 head/tail segment | `artifacts/features/features_embedding_segment_multiseed_qwen14_v1/fakespot_like` | 776 | final segment branch |
| embedding pool | `artifacts/features/features_embedding_pool_v1/fakespot_like` | 518 | used in the no-segment base |
| external reproduction full features | `artifacts/features/features_external_reproduction/fakespot_like` | 720 | basic, burstiness, probability, scale, transition, strict features |
| cross-dynamics with72 | `artifacts/features/features_cross_dynamics_with72_v1/fakespot_like` | 475 | cross-model loss disagreement plus loss dynamics |

Qwen14 segment feature decomposition:

| Qwen14 segment subfamily | count |
|---|---:|
| head-window projected/norm/cosine features | 387 |
| tail-window projected/norm/cosine features | 387 |
| head-tail scalar geometry | 2 |
| total | 776 |

Equivalently, the Qwen14 segment features contain 768 random-projection dimensions and 8 scalar geometry summaries. The segment extraction settings are:

```text
frozen encoder = Qwen2.5-14B
head/tail max_length = 256 tokens
pooling = first token + mask-aware mean
random projection dimension = 64 per pool per seed
projection seeds = 20260525, 20260526, 20260527
dtype = bfloat16
```

## Feature Normalization and Missing Values

For shallow probes trained by `scripts/train_clean_probe_from_features.py`:

| probe family | preprocessing |
|---|---|
| LogisticRegression probes | `SimpleImputer(strategy="median")` then `StandardScaler()` fitted on train only |
| SGDClassifier probes | `SimpleImputer(strategy="median")` then `StandardScaler()` fitted on train only |
| HistGradientBoosting probes | `SimpleImputer(strategy="median")`; no standard scaling |
| ExtraTrees probes | `SimpleImputer(strategy="median")`; no standard scaling |

Final Qwen14 candidate used by the top-level fusion is:

```text
embedding_segment_qwen14_probe__sgd_a1e4
preprocessing = median imputation + standard scaling
```

Top-level fusion normalization is score-rank normalization (`rank01`) because the selected row is `__rank`.

Missing feature values are handled by train-fitted median imputation in the shallow probe pipelines. Missing prediction rows are not allowed: validation/all_samples candidate scores are merged by `id` with one-to-one validation and a row mismatch raises an error.

## Class Imbalance Handling

The train and validation splits are AI-heavy. Class imbalance is handled in the shallow linear/tree probes through `class_weight="balanced"` where supported:

| model | class imbalance handling |
|---|---|
| LogisticRegression | `class_weight="balanced"` |
| SGDClassifier | `class_weight="balanced"` |
| ExtraTreesClassifier | `class_weight="balanced"` |
| HistGradientBoostingClassifier | no class-weight argument in the current script |
| top-level score fusion | no reweighting; validation objective includes ranking and low-FPR terms |

## Threshold Policy

Thresholded metrics use a fixed threshold of `0.5` on the selected score. The dev set is used to select model/candidate/fusion alpha, but not to tune a separate reporting threshold.

The reported `F1@0.5`, accuracy, precision, recall, specificity, Brier score, and ECE use this fixed 0.5 threshold or raw selected scores as appropriate.

## Calibration

No probability calibration is used in the current final model. There is no Platt scaling, isotonic regression, temperature scaling, or dev-calibrated threshold in the final artifact.

Calibration-related metrics such as ECE and Brier score are diagnostic only.

## Hyperparameters

Final top-level fusion:

```text
script = scripts/incremental_blend_with_probe.py
candidate transforms = raw, rank
selected transform = rank
alpha grid = 0.000 to 0.500, step 0.001
selected alpha = 0.108
selection policy = val_composite
```

Qwen14 segment candidate probe used in the final fusion:

```text
model = SGDClassifier
loss = log_loss
alpha = 1e-4
max_iter = 3000
class_weight = balanced
random_state = 7
preprocessing = SimpleImputer(median) + StandardScaler
n_features = 776
```

Other Qwen14 segment probes trained for candidate selection/audit:

```text
LogisticRegression(C=0.3, max_iter=3000, class_weight=balanced, n_jobs=16)
LogisticRegression(C=1.0, max_iter=3000, class_weight=balanced, n_jobs=16)
SGDClassifier(loss=log_loss, alpha=1e-4, max_iter=3000, class_weight=balanced, random_state=7)
SGDClassifier(loss=log_loss, alpha=3e-5, max_iter=3000, class_weight=balanced, random_state=11)
HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, l2_regularization=0.1, random_state=13)
ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight=balanced, n_jobs=24, random_state=17)
```

## Random Seeds and Determinism

The final top-level blend is deterministic once prediction files are fixed. The selected Qwen14 segment candidate has fixed `random_state=7`. Segment random projections use fixed seeds `20260525`, `20260526`, and `20260527`.

The training run trains multiple shallow candidates per feature family, but the final strict artifact uses one selected candidate score from the Qwen14 branch plus one fixed no-segment base score.

## How Many Models Are Trained?

For each feature root, `train_clean_probe_from_features.py` trains six shallow candidate probes. For the final Qwen14 segment branch, these six candidates were trained and evaluated on validation. The final top-level blend then selected one candidate:

```text
selected fusion candidate = embedding_segment_qwen14_probe__sgd_a1e4__rank
```

So the final deployed score formula uses one selected Qwen14 candidate, but it was selected from a candidate pool of six shallow probes and two score transforms (`raw`, `rank`) over the alpha grid.

## Final Metrics

Final strict external `all_samples` metrics:

| metric | value |
|---|---:|
| AUROC | 0.907822 |
| AUPRC | 0.926497 |
| TPR@FPR<=5% | 0.713333 |
| F1@0.5 | 0.838926 |
| Accuracy@0.5 | 0.840000 |
| MCC@0.5 | 0.680060 |
| Brier score | 0.131683 |
| ECE | 0.113796 |

## Important Wording

Use this wording in the manuscript:

> The final detector is a validation-selected score-level fusion of a no-segment evidence base and a Qwen2.5-14B head/tail segment probe. The segment probe is trained as a lightweight shallow classifier on frozen segment features; the final fusion is a deterministic rank-normalized convex blend selected on validation. Thresholded metrics are reported at a fixed threshold of 0.5, and no probability calibration is applied.

Avoid implying that the final fusion is a single end-to-end neural model trained over all raw evidence features.
