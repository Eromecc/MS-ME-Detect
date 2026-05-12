# MS-ME-Detect

Multi-Scale Multi-Evidence Detection of LLM-generated Text.

MS-ME-Detect is a Python project for AI-generated and AI-polished text detection, with a current focus on Chinese and Chinese-English mixed text. It is not an LLM-as-judge detector. Instead, it extracts handcrafted and local language-model features, then trains conventional classifiers for the final decision.

This repository does not include private datasets, model weights, Hugging Face or ModelScope caches, training outputs, or logs. You must prepare your own dataset and local model files before running the full pipeline.

## Current Released Features

The current released feature groups are:

- Statistical features
- Structural features
- Rule-based perturbation features
- Qwen2.5-1.5B probability features

The codebase also includes optional multi-scale probability, scale-response, and binoculars-style modules for extended experiments.

## Method Overview

- Statistical burstiness: sentence length variation, punctuation ratios, type-token ratio, repetition, compression, and Zipf deviation.
- Multi-scale Qwen2.5 probability features: Base models compute PPL and token-level negative log-likelihood distributions.
- Multi-scale probability response features: slopes, gaps, ratios, response areas, and curvature across Qwen2.5 model scales.
- Binoculars-inspired contrast: simplified dual-model contrast between observer and performer Base models. This is inspired by the Binoculars idea, not a full reproduction.
- Structural patterns: template phrase ratios, N-gram repetition, POS ratios, and information-density signals.
- Optional perturbation stability: rule-based perturbations by default, with optional Qwen2.5 Instruct rewriting.

Final outputs include `Yes/No`, AI probability, risk level, and interpretable evidence. Treat the result as a risk assessment, not proof.

## Model Choices

Probability and PPL features use Qwen2.5 Base models:

- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-14B`
- optional `Qwen/Qwen2.5-32B` with `--include_32b`

Instruct models are only used for optional perturbation generation or natural-language explanation:

- `Qwen/Qwen2.5-14B-Instruct`
- optional `Qwen/Qwen2.5-32B-Instruct`

All model names are configurable in `src/config.py` and overridable from the CLI.

## Preparing Qwen2.5 Models

The demo mode does not download or load large Qwen models. It only tests the pipeline. Probability and scale-response features require local Qwen2.5 Base model weights.

Install download tools:

```bash
pip install -U huggingface_hub transformers accelerate safetensors
```

Set cache and model paths:

```bash
source scripts/env.sh
```

Download the minimum Base models for probability features:

```bash
python scripts/download_models.py --models small medium large --backend hf
```

Equivalent through `main.py`:

```bash
python main.py --mode download_models --models small medium large
```

If Hugging Face is not accessible, use ModelScope:

```bash
pip install -U modelscope
python scripts/download_models.py --models small medium large --backend modelscope
```

Check local model readiness:

```bash
python scripts/check_models.py --models small medium large
python main.py --mode check_models --models small medium large
```

### Running long downloads with tmux

Use tmux for long model downloads and monitoring so jobs survive client interruptions.

```bash
tmux ls
tmux attach -t qwen_download
# Detach from tmux: Ctrl-b then d

bash scripts/tmux_monitor_models.sh
bash scripts/tmux_download_models.sh qwen_download medium large
```

Do not start duplicate downloads if `ps` shows an active `modelscope` or `hf download` process for the same model.

Run two-A100 probability extraction from local paths:

```bash
bash scripts/run_probability_features.sh
```

Sequential probability extraction through `main.py`:

```bash
python main.py --mode probability --models small medium large --dtype bfloat16
```

Optional 32B:

```bash
python scripts/download_models.py --models xl --backend hf
```

```bash
CUDA_VISIBLE_DEVICES=1 python src/feature_probability.py \
  --model_key xl \
  --input data/dataset.csv \
  --output features/probability_qwen25_32b.csv \
  --dtype bfloat16 \
  --max_length 1024 \
  --local_files_only
```

Generate scale-response features after probability files exist:

```bash
python src/feature_scale_response.py \
  --feature_dir features \
  --output features/scale_response_features.csv
```

Merge and train:

```bash
python src/merge_features.py
python src/train_eval.py
```

## Multi-Scale Probability Response Features

This module models how the same text's perplexity and token-level loss statistics change as the model scale increases from Qwen2.5-1.5B to 7B, 14B, and optionally 32B. The resulting slopes, gaps, ratios, response areas, and curvature features are used as explainable evidence for detecting AI-generated and AI-polished texts.

It consumes existing probability feature files:

- `features/probability_qwen25_1_5b.csv`
- `features/probability_qwen25_7b.csv`
- `features/probability_qwen25_14b.csv`
- optional `features/probability_qwen25_32b.csv`

Run it independently:

```bash
python src/feature_scale_response.py \
  --feature_dir features \
  --output features/scale_response_features.csv
```

Then merge and train:

```bash
python src/merge_features.py
python src/train_eval.py
```

## Installation

```bash
cd MS-ME-Detect
pip install -r requirements.txt
```

`xgboost`, `sentence-transformers`, and `bitsandbytes` are optional in code. If unavailable, the relevant path is skipped gracefully.

## Data Preparation

Expected file: `data/dataset.csv`

Columns:

- `id`: unique text id
- `text`: input text
- `label`: `1` for AI-generated or AI-polished, `0` for human-written
- `type`: Human / AI-generated / AI-polished
- `source`: source model or source name
- `topic`: topic label

Data files are not shipped with this repository. Prepare your own CSV in the format above. If `data/dataset.csv` is missing, `python main.py --mode demo` creates a very small demo dataset for smoke testing only.

## Training

For a lightweight end-to-end training run without large Qwen model weights:

```bash
python main.py --mode demo
```

For a full feature merge and training pass after your feature CSV files are ready:

```bash
python src/merge_features.py
python src/train_eval.py
```

The trained fusion model and evaluation artifacts are written to `results/`.

## Testing and Evaluation

This repository does not currently ship a standalone `tests/` suite. The practical smoke-test and evaluation entry points are:

```bash
python main.py --mode demo
```

```bash
python main.py --mode check_models --models small medium large
```

```bash
python src/train_eval.py --input features/all_features.csv --result_dir results
```

Use `demo` for a CPU smoke test, `check_models` to validate local Qwen availability, and `train_eval.py` to evaluate a merged feature matrix.

## Running Individual Feature Groups

You can generate one feature family at a time before merging them:

Statistical features:

```bash
python src/feature_burstiness.py --input data/dataset.csv --output features/burstiness_features.csv
```

Structural features:

```bash
python src/feature_structure.py --input data/dataset.csv --output features/structure_features.csv
```

Rule-based perturbation features:

```bash
python src/feature_perturbation.py --input data/dataset.csv --output features/perturbation_features.csv --mode rule
```

Qwen2.5-1.5B probability features:

```bash
python src/feature_probability.py --model_key small --input data/dataset.csv --output features/probability_qwen25_1_5b.csv --local_files_only
```

## Current Feature-Group Ablation

Run ablations on the feature files currently available in `features/all_features.csv`. This utility compares burstiness, structure, perturbation, Qwen2.5-1.5B probability features, selected combinations, and all current numeric features with the same stratified train/test split.

It does not require Qwen2.5-7B, Qwen2.5-14B, or scale-response features.

```bash
python src/group_ablation_current.py
python main.py --mode current_ablation
```

Outputs:

- `results/current_group_ablation_results.csv`
- `results/current_group_ablation_report.txt`

## Fusion Feature Model

The default fusion workflow is:

1. Generate one or more feature CSV files under `features/`
2. Merge them into `features/all_features.csv`
3. Train and evaluate the downstream classifier

Commands:

```bash
python src/merge_features.py
python src/train_eval.py
```

## Full Pipeline

Default full mode attempts Qwen2.5 1.5B, 7B, and 14B Base probability extraction, Binoculars-style features, merge, and training:

```bash
python main.py --mode all --dtype bfloat16 --max_length 1024
```

High-resource local Qwen pipeline using downloaded model keys:

```bash
python main.py --mode full_qwen \
  --models small medium large \
  --dtype bfloat16 \
  --max_length 1024
```

Include 32B only when explicitly requested:

```bash
python main.py --mode full_qwen --models small medium large xl --dtype bfloat16
```

Run only scale-response extraction from existing probability files:

```bash
python main.py --mode scale_response
```

Include 32B:

```bash
python main.py --mode all --include_32b --dtype bfloat16 --device_map auto
```

If a model cannot be loaded, the module writes NaN probability features with warnings and continues.

## Two A100 Probability Extraction

Recommended mode: independent jobs, one model per GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python src/feature_probability.py \
  --model_key small \
  --input data/dataset.csv \
  --output features/probability_qwen25_1_5b.csv \
  --dtype bfloat16 \
  --max_length 1024 \
  --local_files_only
```

```bash
CUDA_VISIBLE_DEVICES=1 python src/feature_probability.py \
  --model_key medium \
  --input data/dataset.csv \
  --output features/probability_qwen25_7b.csv \
  --dtype bfloat16 \
  --max_length 1024 \
  --local_files_only
```

Large model automatic device mapping:

```bash
python src/feature_probability.py \
  --model Qwen/Qwen2.5-32B \
  --input data/dataset.csv \
  --output features/probability_qwen25_32b.csv \
  --dtype bfloat16 \
  --device_map auto \
  --max_length 1024
```

Optional 4-bit:

```bash
python src/feature_probability.py --model Qwen/Qwen2.5-32B --device_map auto --load_4bit
```

## Training Outputs

Typical outputs written under `results/`:

- `results/metrics.csv`
- `results/ablation_results.csv`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/feature_importance.csv`
- `results/feature_importance.png`
- `results/predictions.csv`
- `results/best_model.pkl`
- `results/feature_columns.json`

## Prediction

Lightweight prediction:

```bash
python main.py --mode predict --text "待检测文本"
```

Prediction with local LM features:

```bash
python main.py --mode predict --text "待检测文本" \
  --use_lm_features \
  --predict_model Qwen/Qwen2.5-1.5B \
  --dtype bfloat16
```

Direct script:

```bash
python src/predict.py --text "综上所述，该方法具有重要意义。"
```

## Output Interpretation

- `prediction`: `Yes` means likely AI-generated or AI-polished; `No` means likely human-written.
- `ai_probability`: classifier probability for label `1`.
- `risk_level`: Low `<0.35`, Medium `0.35-0.70`, High `>=0.70`.
- `top_evidence`: simple evidence derived from feature values and feature importance.

## Limitations

- AI text detection is not definitive.
- Human formal writing may be misclassified as AI-like.
- AI-polished text is harder than pure AI-generated text.
- Model probabilities depend on the local model family and domain match.
- Outputs should be interpreted as risk assessment, not absolute proof.
