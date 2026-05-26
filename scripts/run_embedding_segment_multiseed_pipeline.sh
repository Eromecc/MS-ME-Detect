#!/usr/bin/env bash
set -euo pipefail

cd /vepfs-mlp2/queue010/20252203113/MS-ME-Detect

PY=/vepfs-mlp2/queue010/20252203113/conda_envs/scverse-py312/bin/python
DEBERTA_MODEL=/vepfs-mlp2/queue010/20252203113/models/microsoft__deberta-v3-large
ROBERTA_MODEL=/vepfs-mlp2/queue010/20252203113/models/roberta-large

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) build_embedding_segment_multiseed_start"

(
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/build_embedding_segment_features.py \
    --models "deberta=${DEBERTA_MODEL}" \
    --output_dir features_embedding_segment_multiseed_deberta_v1 \
    --batch_size 1024 \
    --max_length 256 \
    --proj_dim 64 \
    --seeds 20260525 20260526 20260527
) &
pid_deberta=$!

(
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/build_embedding_segment_features.py \
    --models "roberta=${ROBERTA_MODEL}" \
    --output_dir features_embedding_segment_multiseed_roberta_v1 \
    --batch_size 3072 \
    --max_length 256 \
    --proj_dim 64 \
    --seeds 20260525 20260526 20260527
) &
pid_roberta=$!

wait "$pid_deberta"
wait "$pid_roberta"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) combine_embedding_segment_multiseed"
"$PY" scripts/combine_feature_roots.py \
  --roots deberta=features_embedding_segment_multiseed_deberta_v1 roberta=features_embedding_segment_multiseed_roberta_v1 \
  --output_dir features_embedding_segment_multiseed_v1

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) train_embedding_segment_multiseed"
"$PY" scripts/train_clean_probe_from_features.py \
  --feature_root features_embedding_segment_multiseed_v1 \
  --feature_name all_features.csv \
  --output_dir results_embedding_segment_multiseed_probe_clean_v1 \
  --model_prefix embedding_segment_multiseed_probe

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) blend_embedding_segment_multiseed"
"$PY" scripts/incremental_blend_with_probe.py \
  --base_name direct_weight_grid_focused_v1 \
  --base_val results_direct_weight_grid_focused_v1/val_composite__validation_predictions.csv \
  --base_all results_direct_weight_grid_focused_v1/val_auroc__all_samples_predictions.csv \
  --candidate_dir results_embedding_segment_multiseed_probe_clean_v1/predictions \
  --output_dir results_incremental_current_best_plus_embedding_segment_multiseed_alpha05_v1 \
  --alpha_max 0.5 \
  --alpha_step 0.001 \
  --plateau_eps 0.00002 0.00005 0.0001 0.0002

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) blend_embedding_segment_multiseed_raw"
"$PY" scripts/incremental_blend_with_probe.py \
  --base_name direct_weight_grid_focused_v1 \
  --base_val results_direct_weight_grid_focused_v1/val_composite__validation_predictions.csv \
  --base_all results_direct_weight_grid_focused_v1/val_auroc__all_samples_predictions.csv \
  --candidate_dir results_embedding_segment_multiseed_probe_clean_v1/predictions \
  --output_dir results_incremental_current_best_plus_embedding_segment_multiseed_raw_v1 \
  --alpha_max 0.5 \
  --alpha_step 0.001 \
  --transforms raw \
  --plateau_eps 0.00002 0.00005 0.0001 0.0002

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done_embedding_segment_multiseed"
