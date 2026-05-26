# Fakespot-Like External Reproduction Snapshot

This document records the current GitHub handoff state for the Fakespot-like reproduction line. It is intentionally GitHub-safe: it describes the data, commands, models, and metrics, but does not publish raw text, local model weights, full feature matrices, or per-sample prediction CSVs.

## Current Best Result

Current best external `all_samples` result:

| Selection file | AUROC | AUPRC | TPR@FPR<=5% | F1@0.5 |
|---|---:|---:|---:|---:|
| `results_incremental_embedding_best_plus_embedding_segment_qwen14_v1/val_composite__all_samples_predictions.csv` | 0.907822 | 0.926497 | 0.713333 | 0.838926 |

Previous best before Qwen14 segment features:

| Selection file | AUROC | AUPRC | TPR@FPR<=5% | F1@0.5 |
|---|---:|---:|---:|---:|
| `results_incremental_current_best_plus_embedding_pool_raw_plateau_v1/val_low_fpr__all_samples_predictions.csv` | 0.899289 | 0.918752 | 0.680000 | 0.784615 |

The improvement is +0.008533 AUROC on `all_samples`. The result above is selected by validation-only policy; `all_samples` labels are used only for final reporting.

## Data Used

Training and validation use prepared Fakespot-like reproduction splits:

| Split | File | Rows | Label distribution |
|---|---|---:|---|
| train | `data/reproduction_datasets/fakespot_like_train.csv` | 132,614 | `{1: 88,694, 0: 43,920}` |
| validation | `data/reproduction_datasets/fakespot_like_val.csv` | 14,932 | `{1: 9,901, 0: 5,031}` |
| external evaluation | `data/test/all_samples_prepared.csv` | 300 | `{1: 150, 0: 150}` |

`all_samples` is treated as an external target set. It is not used to train probes, choose candidate models, choose alpha weights, choose transforms, choose thresholds, calibrate models, or decide source weighting. All candidate and alpha selection is done on `fakespot_like_val.csv`.

The public GitHub repo should not include the raw data CSVs above. They are local/private artifacts or generated datasets. The repo should include the scripts and this reproducibility description.

## Model And Feature Framework

The current line uses frozen encoder/decoder representations as classical features:

1. For each text, run a local frozen model twice with `max_length=256`:
   - head window: tokenizer truncation side `right`
   - tail window: tokenizer truncation side `left`
2. Extract CLS/first-token and attention-mask mean pooled hidden states from both windows.
3. Add scalar diagnostics:
   - head CLS norm
   - head mean norm
   - head CLS/mean cosine
   - tail CLS norm
   - tail mean norm
   - tail CLS/mean cosine
   - head/tail mean cosine
   - head/tail mean L2 distance
4. Add deterministic random projection features for CLS and mean vectors using seeds `20260525`, `20260526`, and `20260527`, with `proj_dim=64`.
5. Train clean probes on train, select on validation, report `all_samples` only as final external evaluation.
6. Blend the probe scores into the current best base score with validation-selected alpha.

Important scripts:

- `scripts/build_embedding_pool_features.py`
- `scripts/build_embedding_segment_features.py`
- `scripts/train_clean_probe_from_features.py`
- `scripts/incremental_blend_with_probe.py`
- `scripts/combine_feature_roots.py`
- `scripts/run_embedding_segment_multiseed_pipeline.sh`
- `scripts/run_embedding_segment_extra_followup.sh`
- `scripts/run_embedding_segment_llama_followup.sh`
- `scripts/run_embedding_segment_qwen7_followup.sh`

## Models Tested In This Line

Completed or running local models:

| Model | Local path | Status |
|---|---|---|
| DeBERTa-v3-large | `/vepfs-mlp2/queue010/20252203113/models/microsoft__deberta-v3-large` | completed segment features; did not improve current best |
| RoBERTa-large | `/vepfs-mlp2/queue010/20252203113/models/roberta-large` | completed segment features; did not improve current best |
| Qwen2.5-14B | `/vepfs-mlp2/queue010/20252203113/models/Qwen2.5-14B` | completed segment features; produced current best 0.907822 AUROC |
| Mistral-7B-v0.1 | `/vepfs-mlp2/queue010/20252203113/models/mistralai__Mistral-7B-v0.1` | completed segment features; queued for combined follow-up |
| Llama-3.1-8B | `/vepfs-mlp2/queue010/20252203113/models/meta-llama__Meta-Llama-3.1-8B` | running segment features |
| Qwen2.5-7B | `/vepfs-mlp2/queue010/20252203113/models/Qwen2.5-7B` | running segment features |

Qwen2.5-32B and Qwen2.5-72B are not the current priority for segment features because their single-GPU cost is much higher. They remain possible follow-up candidates if smaller complementary models stop improving the validation-selected external result.

## Commands For The Current Reproduction Line

Build DeBERTa/RoBERTa segment features, train probes, and run the first fusion:

```bash
bash scripts/run_embedding_segment_multiseed_pipeline.sh
```

Build Qwen14 and Mistral segment features on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/build_embedding_segment_features.py \
  --models qwen14=/path/to/models/Qwen2.5-14B \
  --output_dir features_embedding_segment_multiseed_qwen14_v1 \
  --batch_size 512 \
  --max_length 256 \
  --proj_dim 64 \
  --seeds 20260525 20260526 20260527

CUDA_VISIBLE_DEVICES=0 python scripts/build_embedding_segment_features.py \
  --models mistral=/path/to/models/mistralai__Mistral-7B-v0.1 \
  --output_dir features_embedding_segment_multiseed_mistral_v1 \
  --batch_size 768 \
  --max_length 256 \
  --proj_dim 64 \
  --seeds 20260525 20260526 20260527
```

Automatically train and blend Qwen14, Mistral, and combined feature roots:

```bash
bash scripts/run_embedding_segment_extra_followup.sh
```

Additional running follow-ups:

```bash
bash scripts/run_embedding_segment_llama_followup.sh
bash scripts/run_embedding_segment_qwen7_followup.sh
```

## Why Qwen14 Helped

The previous best was already a validation-selected blend using pooled embedding features. DeBERTa/RoBERTa segment features alone did not transfer well to `all_samples`; their validation scores were very high but external AUROC dropped, which indicates train/validation over-specialization.

Qwen14 segment features appear more complementary to the existing base score. The likely reasons are:

- Decoder-only Qwen hidden states encode generation-style regularities differently from encoder-only DeBERTa/RoBERTa embeddings.
- Head/tail pooling captures beginning/end distribution shifts that whole-text pooling can smooth out.
- Random projections preserve broad hidden-state geometry while keeping a tabular probe small enough for robust validation-only selection.
- The final gain comes from blending with the previous best, not from replacing it. The Qwen14 probe alone is not treated as the final detector; it adds complementary ranking signal.

This is still an empirical external-target result. The correct claim is: under the current validation-only selection protocol, Qwen14 head/tail segment features improve the local `all_samples` AUROC from 0.899289 to 0.907822. It is not a guarantee of 0.90+ on unrelated future data.

## GitHub Upload Policy

Commit these to GitHub:

- source code under `scripts/` and `src/`
- `README.md`, `PROJECT_STATE.md`, and `docs/*.md`
- small manifests and aggregate result summaries if needed

Do not commit these:

- local model directories under `/vepfs-mlp2/.../models`
- `data/` raw/prepared CSVs unless licensing and privacy are verified
- `features_embedding_*/*/all_features.csv`
- full validation/all_samples prediction CSVs
- checkpoint `.joblib`, `.pt`, `.safetensors`, caches, logs, and tmux output

The files above are large and may contain raw text, labels, IDs, or model-derived private artifacts. They should remain local and be regenerated with the scripts.
