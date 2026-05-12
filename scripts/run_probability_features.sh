#!/usr/bin/env bash
set -euo pipefail

cd /vepfs-mlp2/queue010/20252203113/MS-ME-Detect

export HF_HOME=/vepfs-mlp2/queue010/20252203113/hf_cache
export TRANSFORMERS_CACHE=/vepfs-mlp2/queue010/20252203113/hf_cache

CUDA_VISIBLE_DEVICES=0 python src/feature_probability.py \
  --model_key small \
  --input data/dataset.csv \
  --output features/probability_qwen25_1_5b.csv \
  --dtype bfloat16 \
  --max_length 1024 \
  --local_files_only

CUDA_VISIBLE_DEVICES=1 python src/feature_probability.py \
  --model_key medium \
  --input data/dataset.csv \
  --output features/probability_qwen25_7b.csv \
  --dtype bfloat16 \
  --max_length 1024 \
  --local_files_only

CUDA_VISIBLE_DEVICES=0 python src/feature_probability.py \
  --model_key large \
  --input data/dataset.csv \
  --output features/probability_qwen25_14b.csv \
  --dtype bfloat16 \
  --max_length 1024 \
  --local_files_only

# Optional 32B:
# CUDA_VISIBLE_DEVICES=1 python src/feature_probability.py \
#   --model_key xl \
#   --input data/dataset.csv \
#   --output features/probability_qwen25_32b.csv \
#   --dtype bfloat16 \
#   --max_length 1024 \
#   --local_files_only

python src/feature_scale_response.py \
  --feature_dir features \
  --output features/scale_response_features.csv

python src/merge_features.py
python src/train_eval.py

