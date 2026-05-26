#!/usr/bin/env bash
set -euo pipefail
cd /vepfs-mlp2/queue010/20252203113/MS-ME-Detect

PY=/vepfs-mlp2/queue010/20252203113/conda_envs/scverse-py312/bin/python
BASE_NAME=embedding_pool_raw_plateau_v1
BASE_VAL=results_incremental_current_best_plus_embedding_pool_raw_plateau_v1/val_low_fpr__validation_predictions.csv
BASE_ALL=results_incremental_current_best_plus_embedding_pool_raw_plateau_v1/val_low_fpr__all_samples_predictions.csv

wait_manifest() {
  while [[ ! -f "$1" ]]; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting $1"
    sleep 60
  done
}

train_and_blend() {
  local feature_root=$1 prefix=$2 probe_dir=$3 out_rank=$4 out_raw=$5
  if [[ ! -f "$probe_dir/clean_probe_summary.csv" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) train $feature_root"
    "$PY" scripts/train_clean_probe_from_features.py \
      --feature_root "$feature_root" \
      --feature_name all_features.csv \
      --output_dir "$probe_dir" \
      --model_prefix "$prefix"
  fi
  if [[ ! -f "$out_rank/incremental_probe_selected.csv" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) blend rank $feature_root"
    "$PY" scripts/incremental_blend_with_probe.py \
      --base_name "$BASE_NAME" \
      --base_val "$BASE_VAL" \
      --base_all "$BASE_ALL" \
      --candidate_dir "$probe_dir/predictions" \
      --output_dir "$out_rank" \
      --alpha_max 0.5 \
      --alpha_step 0.001 \
      --plateau_eps 0.00002 0.00005 0.0001 0.0002
  fi
  if [[ ! -f "$out_raw/incremental_probe_selected.csv" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) blend raw $feature_root"
    "$PY" scripts/incremental_blend_with_probe.py \
      --base_name "$BASE_NAME" \
      --base_val "$BASE_VAL" \
      --base_all "$BASE_ALL" \
      --candidate_dir "$probe_dir/predictions" \
      --output_dir "$out_raw" \
      --alpha_max 0.5 \
      --alpha_step 0.001 \
      --transforms raw \
      --plateau_eps 0.00002 0.00005 0.0001 0.0002
  fi
}

wait_manifest features_embedding_segment_multiseed_tinyllama_v1/embedding_segment_manifest.json
train_and_blend \
  features_embedding_segment_multiseed_tinyllama_v1 \
  embedding_segment_tinyllama_probe \
  results_embedding_segment_tinyllama_probe_clean_v1 \
  results_incremental_embedding_best_plus_embedding_segment_tinyllama_v1 \
  results_incremental_embedding_best_plus_embedding_segment_tinyllama_raw_v1

wait_manifest features_embedding_segment_multiseed_qwen14_v1/embedding_segment_manifest.json
wait_manifest features_embedding_segment_multiseed_mistral_v1/embedding_segment_manifest.json
if [[ ! -f features_embedding_segment_multiseed_tinyllama_qwen14_mistral_v1/combine_manifest.json ]]; then
  "$PY" scripts/combine_feature_roots.py \
    --roots tinyllama=features_embedding_segment_multiseed_tinyllama_v1 qwen14=features_embedding_segment_multiseed_qwen14_v1 mistral=features_embedding_segment_multiseed_mistral_v1 \
    --output_dir features_embedding_segment_multiseed_tinyllama_qwen14_mistral_v1
fi
train_and_blend \
  features_embedding_segment_multiseed_tinyllama_qwen14_mistral_v1 \
  embedding_segment_tinyllama_qwen14_mistral_probe \
  results_embedding_segment_tinyllama_qwen14_mistral_probe_clean_v1 \
  results_incremental_embedding_best_plus_embedding_segment_tinyllama_qwen14_mistral_v1 \
  results_incremental_embedding_best_plus_embedding_segment_tinyllama_qwen14_mistral_raw_v1

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done_tinyllama_segment_followup"
