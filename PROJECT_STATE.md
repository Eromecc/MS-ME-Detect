# PROJECT_STATE

Updated at: 2026-05-17T23:36:03Z

## 0. Final full loss ablation status

The full strict Text-Koopman loss ablation completed. The tmux session
`strict_loss_update_full` is no longer present and no
`run_text_koopman_strict_math_experiment.py` process is running.

Final manifest:

- `results_text_koopman_strict_math/loss_update/loss_update_manifest.json`
- completed at: `2026-05-17T16:10:48.648046+00:00`
- experiments: `6`
- errors: `0`
- `error_log.txt`: not present

Completed configurations:

- `recon_only`, rank 16: ok
- `recon_only`, rank 32: ok
- `recon_lin`, rank 16: ok
- `recon_lin`, rank 32: ok
- `recon_lin_multi`, rank 16: ok
- `recon_lin_multi`, rank 32: ok

Key outputs:

- `results_text_koopman_strict_math/loss_update/loss_ablation_summary.csv`
- `results_text_koopman_strict_math/loss_update/all_samples_summary.csv`
- `results_text_koopman_strict_math/loss_update/loss_curves.csv`
- `results_text_koopman_strict_math/loss_update/LOSS_UPDATE_REPORT.md`
- `results_text_koopman_strict_math/loss_update/plots/`

Leakage audit passed:

- `passed=true`
- `uses_global_K_parameter=false`
- `uses_label_classifier=false`
- `uses_projection=false`
- `banned_feature_names=[]`
- `non_strict_feature_names=[]`

All-samples strict spectral-only summary:

```text
recon_only rank16:       AUROC 0.4706, AUPRC 0.4750, TPR@FPR5% 0.0200
recon_only rank32:       AUROC 0.5422, AUPRC 0.5520, TPR@FPR5% 0.0800
recon_lin rank16:        AUROC 0.5350, AUPRC 0.5728, TPR@FPR5% 0.1133
recon_lin rank32:        AUROC 0.5453, AUPRC 0.5670, TPR@FPR5% 0.0800
recon_lin_multi rank16:  AUROC 0.5226, AUPRC 0.5404, TPR@FPR5% 0.0733
recon_lin_multi rank32:  AUROC 0.5274, AUPRC 0.5481, TPR@FPR5% 0.1000
```

All-samples `full_plus_transition_plus_strict_koopman` summary:

```text
recon_only rank16:       AUROC 0.7120, AUPRC 0.6860, TPR@FPR5% 0.1400
recon_only rank32:       AUROC 0.6947, AUPRC 0.6702, TPR@FPR5% 0.1400
recon_lin rank16:        AUROC 0.6856, AUPRC 0.6709, TPR@FPR5% 0.1133
recon_lin rank32:        AUROC 0.6577, AUPRC 0.6564, TPR@FPR5% 0.2067
recon_lin_multi rank16:  AUROC 0.6632, AUPRC 0.6498, TPR@FPR5% 0.1467
recon_lin_multi rank32:  AUROC 0.6438, AUPRC 0.6333, TPR@FPR5% 0.0800
```

Interpretation:

- `L_lin` improved strict spectral-only versus `recon_only/rank16`, especially
  AUPRC and TPR@FPR5%.
- `L_multi` did not further improve strict spectral-only in this full ablation.
- Best all-samples combined AUROC/AUPRC was `recon_only/rank16`, not the new
  dynamics-loss variants.
- Best all-samples combined TPR@FPR5% was `recon_lin/rank32` at `0.2067`.
- No NaN/OOM/DMD fallback instability observed.
- Recommendation in `docs/STRICT_TEXT_KOOPMAN_LOSS_UPDATE_AUDIT.md` was updated:
  do not launch a larger targeted run solely from this loss update without a
  decision on whether strictness, low-FPR recall, or AUROC/AUPRC is primary.

Updated at: 2026-05-17T14:53:43Z

## 1. Active full loss ablation run

User confirmed to continue the full strict Text-Koopman loss ablation and not stop early. The run is active in tmux:

```bash
tmux attach -t strict_loss_update_full
```

Command:

```bash
python scripts/run_text_koopman_strict_math_experiment.py \
  --train_sources leave_out_ghostbuster \
  --model qwen25_1_5b \
  --max_length 256 \
  --observable_multiplier 2 \
  --dmd_ranks 16 32 \
  --loss_modes recon_only recon_lin recon_lin_multi \
  --alpha_lin 1.0 \
  --beta_multi 0.5 \
  --multi_steps 2 3 \
  --epochs 5 \
  --run_hidden_cache \
  --run_lifting_train \
  --run_feature_extract \
  --run_eval \
  --run_audit \
  --run_plots \
  --log_loss_terms \
  --resume \
  --output_subdir loss_update \
  --seed 42
```

Current observed status:

- Process PID: `2997428`.
- Log: `results_text_koopman_strict_math/loss_update/full_ablation_resume.log`.
- Partial manifest: `results_text_koopman_strict_math/loss_update/loss_update_manifest.partial.json`.
- Completed manifest entries: 4/6.
  - `recon_only`, rank 16: ok.
  - `recon_only`, rank 32: ok.
  - `recon_lin`, rank 16: ok.
  - `recon_lin`, rank 32: ok.
- `recon_lin_multi`, rank 16 training completed epoch 5 and is currently in feature extraction:
  - latest log line: `feature_extract_start ... dataset=m4_train`.
  - epoch 5: `train_total=0.379151`, `dev_total=0.313449`.
  - `train_lin=0.051055`, `train_multi=0.075183`.
  - `dev_lin=0.046198`, `dev_multi=0.067720`.
  - `valid_multi_steps=2`, `skipped_multi_steps=0`, `dmd_fallbacks=0`.
- `recon_lin_multi`, rank 32 has not started yet.
- Errors observed: none. `error_log.txt` not present at last check.
- Warnings observed: only `torch.load(weights_only=False)` FutureWarnings.
- GPU use: CUDA visible with two A100s, current code uses GPU0; GPU1 remains idle. No multi-GPU refactor was attempted.

Next minimal plan:

1. Keep monitoring the active tmux run.
2. Wait for `recon_lin_multi` rank 16 feature/eval/audit to finish and for manifest entry 5/6.
3. Then monitor `recon_lin_multi` rank 32 training, feature extraction, eval, audit, and plots.
4. After the process exits, inspect:
   - `loss_ablation_summary.csv`
   - `all_samples_summary.csv`
   - `loss_curves.csv`
   - `loss_update_manifest.json`
   - `LOSS_UPDATE_REPORT.md`
   - `error_log.txt` if present
5. Perform the final requirement audit before reporting.

Updated at: 2026-05-17T09:25:00Z

## 1. 最新续跑状态

本轮按交接要求先读取了 `PROJECT_STATE.md` 和当前 git diff，没有递归扫描禁扫目录。随后补了两个长任务可靠性修正：

- `src/text_koopman_strict_math_train.py`
  - `batch_loss` 不再为了 variance loss 二次调用 `model(h)`；`compute_strict_koopman_losses(..., include_z=True)` 返回当前 forward 的 `z`。
  - `train_strict_math_lifting(..., resume=True)` 支持从 `strict_math_lifting_last.pt` 恢复。
  - 每个 epoch 写 `strict_math_lifting_last.pt`，包含 model、optimizer、history、best state、bad epoch count。
  - 每个 epoch 打印 progress line。
- `scripts/run_text_koopman_strict_math_experiment.py`
  - 将 `--resume` 传入训练函数。
  - 每个配置完成或失败后写 `results_text_koopman_strict_math/loss_update/loss_update_manifest.partial.json`。
  - partial manifest 记录配置 elapsed seconds、batch size、loss mode、rank 和 GPU info。

回归验证：

- `python -m py_compile src/text_koopman_strict_math_train.py scripts/run_text_koopman_strict_math_experiment.py` passed.
- Synthetic gradient check still passed:
  - `loss`: `4.771623611450195`
  - `lin`: `1.6365859508514404`
  - `multi`: `2.748098850250244`
  - `g_grad_abs_sum`: `65.18624329566956`
  - `warning`: `None`

Full ablation resume command was started without `--max_rows_per_split`:

```bash
python scripts/run_text_koopman_strict_math_experiment.py \
  --train_sources leave_out_ghostbuster \
  --model qwen25_1_5b \
  --max_length 256 \
  --observable_multiplier 2 \
  --dmd_ranks 16 32 \
  --loss_modes recon_only recon_lin recon_lin_multi \
  --alpha_lin 1.0 \
  --beta_multi 0.5 \
  --multi_steps 2 3 \
  --epochs 5 \
  --run_hidden_cache \
  --run_lifting_train \
  --run_feature_extract \
  --run_eval \
  --run_audit \
  --run_plots \
  --log_loss_terms \
  --resume \
  --output_subdir loss_update \
  --seed 42
```

It was stopped again after the 45-minute handoff threshold.

Progress before stopping:

- Completed full config:
  - `leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256`
  - status: `ok`
  - loss_mode: `recon_only`
  - dmd_rank: `16`
  - train sequences: `16204`
  - dev sequences: `2030`
  - best epoch: `5`
  - elapsed seconds: `2304.9409807920456`
  - no OOM events
- Started next config:
  - `recon_only`, `dmd_rank=32`
  - observed stdout: `[strict-math] epoch=1 loss_mode=recon_only dmd_rank=32 train_total=0.463862 dev_total=0.0790503`
  - explicit resume checkpoint exists:
    `checkpoints_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank32_len256/strict_math_lifting_last.pt`

Top-level output files now present in `results_text_koopman_strict_math/loss_update/`:

- `strict_math_dry_run.json`
- `strict_math_requirement_checklist.csv`
- `leakage_audit.json`
- `loss_update_manifest.partial.json`
- `leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_strict_koopman_spectral_only_classifier_selection.csv`
- `leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_strict_koopman_classifier_selection.csv`
- `leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_transition_plus_strict_koopman_classifier_selection.csv`
- `leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_transition_plus_strict_koopman_classifier_selection.csv`

Partial manifest summary:

```text
experiments:
  leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256:
    status: ok
    checkpoint: checkpoints_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256
    loss_mode: recon_only
    dmd_rank: 16
    batch_size: 1
    elapsed_seconds: 2304.9409807920456
errors: []
gpu_info:
  cuda_available: true
  cuda_device_count: 2
  devices: [NVIDIA A100-SXM4-80GB, NVIDIA A100-SXM4-80GB]
```

Warnings/failures in this continuation:

- Only `torch.load(weights_only=False)` FutureWarnings from hidden cache/checkpoint loading.
- No NaN/OOM observed.
- GPU0 was used; GPU1 remained mostly idle. No multi-GPU refactor was attempted.
- The process was stopped intentionally at the handoff threshold, not due to failure.

Next minimal executable plan:

1. Read this `PROJECT_STATE.md` and inspect current git diff for edited files only.
2. Resume the same full ablation command with `--resume` and no `--max_rows_per_split`.
3. Expect it to resume `recon_only/rank32` from `strict_math_lifting_last.pt`.
4. Continue monitoring top-level `results_text_koopman_strict_math/loss_update/` outputs and `loss_update_manifest.partial.json`.
5. After all 6 configs finish, inspect:
   - `loss_ablation_summary.csv`
   - `all_samples_summary.csv`
   - `loss_curves.csv`
   - `loss_update_manifest.json`
   - `LOSS_UPDATE_REPORT.md`
   - `error_log.txt` if present
6. Then perform the completion audit against the objective before final reporting.

## 1. 当前目标

在 `MS-ME-Detect` 中继续完善 strict mathematical Text-Koopman pipeline 的训练目标：审计原 loss，加入 per-document local DMD 的 `L_lin` 和 `L_multi`，保持 `Qwen hidden states -> learned lifting g_theta -> local DMD K_i -> spectral-only features -> classical classifier`，并运行 full loss ablation。不得使用 `all_samples` 做训练、early stopping、loss tuning、model selection、threshold selection 或 calibration。

## 2. 已完成的步骤

- 完成指定文件的代码审计：
  - `src/text_koopman_strict_math_train.py`
  - `src/text_koopman_strict_math_model.py`
  - `src/text_koopman_strict_math_features.py`
  - `scripts/run_text_koopman_strict_math_experiment.py`
  - `docs/STRICT_MATHEMATICAL_TEXT_KOOPMAN.md`
  - `docs/STRICT_TEXT_KOOPMAN_LEAKAGE_AUDIT.md`
- 写出审计文件：`results_text_koopman_strict_math/loss_update_audit.md`
- 确认原训练不是纯 autoencoder：已有 reconstruction、one-step DMD、fixed multi-step、stability、variance 项。
- 确认原训练的缺陷：没有显式 `compute_strict_koopman_losses` API，没有 CLI loss modes/alpha/beta/multi_steps，DMD 失败会静默置零 dynamics loss。
- 在 `src/text_koopman_strict_math_train.py` 加入：
  - `compute_strict_koopman_losses(...)`
  - `recon_only | recon_lin | recon_lin_multi`
  - per-document local truncated exact DMD `K_tilde_i`
  - differentiable ridge reduced least-squares fallback for SVD RuntimeError
  - configurable `alpha_lin`, `beta_multi`, `multi_steps`, `ridge`, `stability_weight`
  - loss logging fields: `recon`, `lin`, `multi`, `stability`, `var`, `total`, `valid_multi_steps`, `skipped_multi_steps`, `dmd_fallbacks`
- 在 runner 中加入：
  - `--loss_mode`
  - `--loss_modes`
  - `--alpha_lin`
  - `--beta_multi`
  - `--multi_steps`
  - `--disable_stability_loss`
  - `--log_loss_terms`
  - `--resume`
  - `--output_subdir`
  - GPU info manifest logging
  - `error_log.txt` append on per-config failure
  - `loss_ablation_summary.csv`, `all_samples_summary.csv`, `loss_curves.csv`, `loss_update_manifest.json`, `LOSS_UPDATE_REPORT.md`
  - loss-update plots under `results_text_koopman_strict_math/loss_update/plots/`
- Updated docs:
  - `docs/STRICT_MATHEMATICAL_TEXT_KOOPMAN.md`
  - `docs/STRICT_TEXT_KOOPMAN_LOSS_UPDATE_AUDIT.md`
- Verification:
  - `python -m py_compile src/text_koopman_strict_math_train.py scripts/run_text_koopman_strict_math_experiment.py` passed.
  - Synthetic gradient check passed:
    - `loss`: `4.771623611450195`
    - `lin`: `1.6365859508514404`
    - `multi`: `2.748098850250244`
    - `g_grad_abs_sum`: `65.18624329566956`
    - `warning`: `None`
- Dry-run passed and wrote `results_text_koopman_strict_math/loss_update/strict_math_dry_run.json`.
  - hidden size: `1536`
  - observable dim: `3072`
  - GPU: two `NVIDIA A100-SXM4-80GB`

## 3. 修改过的文件

- `src/text_koopman_strict_math_train.py`
- `scripts/run_text_koopman_strict_math_experiment.py`
- `docs/STRICT_MATHEMATICAL_TEXT_KOOPMAN.md`
- `docs/STRICT_TEXT_KOOPMAN_LOSS_UPDATE_AUDIT.md`
- `results_text_koopman_strict_math/loss_update_audit.md`
- `PROJECT_STATE.md`

Unrelated pre-existing untracked file observed and not touched:

- `features/scale_response_manifest.json`

## 4. 关键命令和输出路径

Syntax check:

```bash
python -m py_compile \
  /vepfs-mlp2/queue010/20252203113/MS-ME-Detect/src/text_koopman_strict_math_train.py \
  /vepfs-mlp2/queue010/20252203113/MS-ME-Detect/scripts/run_text_koopman_strict_math_experiment.py
```

Synthetic gradient check:

```bash
python -c "import sys, torch; sys.path.insert(0, '/vepfs-mlp2/queue010/20252203113/MS-ME-Detect'); from src.text_koopman_strict_math_model import StrictKoopmanLifting; from src.text_koopman_strict_math_train import compute_strict_koopman_losses; torch.manual_seed(0); m=StrictKoopmanLifting(8,16); h=torch.randn(12,8); loss,d=compute_strict_koopman_losses(m,h,alpha=1.0,beta=0.5,multi_steps=(2,3),dmd_rank=4,loss_mode='recon_lin_multi'); loss.backward(); g=sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in m.g.parameters()); print({'loss': float(loss.detach()), 'lin': d['lin'], 'multi': d['multi'], 'g_grad_abs_sum': g, 'warning': d['dmd_warning']})"
```

Dry-run/readiness:

```bash
python scripts/run_text_koopman_strict_math_experiment.py \
  --dry_run \
  --train_sources leave_out_ghostbuster \
  --model qwen25_1_5b \
  --max_rows_per_split 50 \
  --loss_mode recon_lin_multi \
  --alpha_lin 1.0 \
  --beta_multi 0.5 \
  --multi_steps 2 3 \
  --seed 42 \
  --output_subdir loss_update
```

Full ablation command was started without `--max_rows_per_split`:

```bash
python scripts/run_text_koopman_strict_math_experiment.py \
  --train_sources leave_out_ghostbuster \
  --model qwen25_1_5b \
  --max_length 256 \
  --observable_multiplier 2 \
  --dmd_ranks 16 32 \
  --loss_modes recon_only recon_lin recon_lin_multi \
  --alpha_lin 1.0 \
  --beta_multi 0.5 \
  --multi_steps 2 3 \
  --epochs 5 \
  --run_hidden_cache \
  --run_lifting_train \
  --run_feature_extract \
  --run_eval \
  --run_audit \
  --run_plots \
  --log_loss_terms \
  --resume \
  --output_subdir loss_update \
  --seed 42
```

It was stopped at the 45-minute handoff threshold before any top-level full-run output appeared. Top-level files currently observed in `results_text_koopman_strict_math/loss_update/`:

- `strict_math_dry_run.json`

## 5. 失败/警告信息

- Default sandbox failed on this host with:
  - `bwrap: No permissions to create a new namespace...`
  - Therefore file reads/checks/runs were executed with approved escalation.
- Dry-run emitted repeated pandas `FutureWarning` from `run_transition_fullscale_optimized.py:83`; not fatal.
- Full ablation emitted a `torch.load(weights_only=False)` future warning from `src/hidden_state_cache.py:73`; not fatal.
- Full ablation process PID was `2919798`; GPU usage before stopping:
  - GPU0: about `3476 MiB`, `47%` utilization
  - GPU1: about `7 MiB`, `0%` utilization
  - Current code used single CUDA device; no multi-GPU refactor was attempted.
- Full ablation was stopped because runtime exceeded the user-defined 45-minute handoff threshold.

## 6. 下一步最小可执行计划

1. Start a new Codex session with the required sentence:
   “请先读取 PROJECT_STATE.md 和当前 git diff，然后继续上一个任务。
   不要扫描 data/、output/、checkpoints/、figures/、logs/、*.h5ad、*.pt、*.pkl、*.log。
   先总结当前状态，再给出下一步计划，确认后再执行。”
2. Inspect current git diff only for edited files.
3. Consider one small code-quality fix before rerun: avoid the second `model(h)` in `batch_loss` by returning `z` or `var` from `compute_strict_koopman_losses`.
4. Resume the full ablation command above with `--resume` and no `--max_rows_per_split`.
5. Monitor top-level outputs:
   - `results_text_koopman_strict_math/loss_update/loss_ablation_summary.csv`
   - `results_text_koopman_strict_math/loss_update/all_samples_summary.csv`
   - `results_text_koopman_strict_math/loss_update/loss_curves.csv`
   - `results_text_koopman_strict_math/loss_update/loss_update_manifest.json`
   - `results_text_koopman_strict_math/loss_update/LOSS_UPDATE_REPORT.md`
   - `results_text_koopman_strict_math/loss_update/error_log.txt`
6. After completion, answer whether `recon_lin` improves strict spectral-only over `recon_only`, whether `recon_lin_multi` further improves, whether full+transition+strict improves TPR@FPR5%, and whether it exceeds the previous transition reference.

---

Updated at: 2026-05-18T00:00:00Z

## all_samples external scoreboard and deployment consistency pass

Completed:

- Added `scripts/run_all_samples_external_scoreboard.py`.
  - Dry-run/readiness mode writes `results_all_samples_scoreboard/readiness_manifest.json`.
  - Trusted full mode loads selected checkpoints, aligns features, reports coverage, evaluates metrics, saves curves, calibration bins, subgroup diagnostics, bootstrap CIs, and error analysis.
  - Safe `--predictions_csv` mode curates existing prediction files without loading `.joblib` pickles.
- Added `scripts/run_threshold_calibration_dev_only.py`.
  - Thresholds and Platt/isotonic calibrators are selected or fit on public prediction files only.
  - `all_samples` is used only for final transfer metrics.
- Patched `src/predict.py`.
  - Adds feature compatibility reporting.
  - Adds `--strict_feature_check`.
  - Refuses confident single-text prediction when more than 20% of required features are missing and prints:
    `This checkpoint requires features not generated by the current prediction pipeline.`
  - Reports missing feature families among `burst`, `struct`, `probability`, `scale_response`, `transition`, and `strict_koopman`.
- Updated `.gitignore`, `docs/GITHUB_ARTIFACT_MANIFEST.md`, and `docs/REPRODUCIBILITY_COMMANDS.md`.
- Generated curated outputs under `results_all_samples_scoreboard/`.

Commands run:

```bash
python scripts/run_all_samples_external_scoreboard.py --dry_run \
  --checkpoint_dirs checkpoints_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256 \
  --feature_files features_external/all_samples_full_allfeatures/all_features.csv features_text_koopman_strict_math/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256/all_samples_strict_koopman_features.csv \
  --output_dir results_all_samples_scoreboard

python scripts/run_all_samples_external_scoreboard.py \
  --predictions_csv results_text_koopman_strict_math/loss_update/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_transition_plus_strict_koopman_to_all_samples/predictions.csv \
  --scoreboard_name leave_out_ghostbuster_recon_only_rank16_full_plus_transition_plus_strict_koopman \
  --output_dir results_all_samples_scoreboard \
  --bootstrap_samples 1000 \
  --seed 42

python scripts/run_threshold_calibration_dev_only.py \
  --dev_predictions \
    results_text_koopman_strict_math/loss_update/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_transition_plus_strict_koopman_to_m4_test/predictions.csv \
    results_text_koopman_strict_math/loss_update/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_transition_plus_strict_koopman_to_ghostbuster_test/predictions.csv \
    results_text_koopman_strict_math/loss_update/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_transition_plus_strict_koopman_to_hc3_plus_test/predictions.csv \
  --all_samples_predictions results_text_koopman_strict_math/loss_update/leave_out_ghostbuster_qwen25_1_5b_recon_only_hidden_to_obs3072_rank16_len256_full_plus_transition_plus_strict_koopman_to_all_samples/predictions.csv \
  --output_dir results_all_samples_scoreboard/threshold_calibration_best_recon_only_rank16
```

Full checkpoint evaluation note:

- Attempted trusted full mode was blocked by the execution policy because
  `joblib.load` on generated checkpoint pickles can execute arbitrary code.
- The code path is implemented, but the run was not executed in this session.
- Safe curated metrics were regenerated from existing prediction CSVs.

Current best all_samples result:

- train source: `leave_out_ghostbuster`
- model/features: `qwen25_1_5b`, `full_plus_transition_plus_strict_koopman`
- loss/rank: `recon_only`, `dmd_rank=16`
- classifier: `RandomForest`
- AUROC `0.711978`
- AUPRC `0.686020`
- F1 `0.678899`
- TPR@FPR1% `0.013333`
- TPR@FPR5% `0.140000`
- TPR@FPR10% `0.320000`
- ECE `0.179622`
- Brier `0.256891`
- MCC `0.158035`

Bootstrap CI from `results_all_samples_scoreboard/.../bootstrap_ci.csv`:

- AUROC mean `0.711595`, 95% CI `[0.653606, 0.768311]`
- AUPRC mean `0.692574`, 95% CI `[0.609083, 0.774596]`
- F1 mean `0.679872`, 95% CI `[0.632069, 0.726059]`
- MCC mean `0.158547`, 95% CI `[0.059103, 0.232511]`
- TPR@FPR5% mean `0.155264`, 95% CI `[0.020968, 0.316992]`
- ECE mean `0.183298`, 95% CI `[0.133341, 0.232292]`
- Brier mean `0.256363`, 95% CI `[0.235286, 0.278347]`

Threshold/calibration transfer:

- Default threshold 0.5: F1 `0.678899`, TPR@FPR5% `0.140000`, FPR `0.920000`.
- Public-dev best-F1 threshold `0.633333`: all_samples F1 `0.674352`, FPR `0.533333`.
- Public-dev FPR 1% threshold `0.94`: all_samples TPR `0.013333`, FPR `0.020000`, F1 `0.025806`.
- Public-dev FPR 5% threshold `0.806667`: all_samples TPR `0.140000`, FPR `0.053333`, F1 `0.234637`.
- Low-FPR transfer mostly collapses in recall; it controls FPR but does not improve practical low-FPR detection.
- Platt calibration improves ECE from `0.179622` to `0.140117` and Brier from `0.256891` to `0.240694` on all_samples.
- Isotonic improves ECE to `0.123918` and Brier to `0.237655`, but AUROC/AUPRC shift down because isotonic maps scores non-strictly monotonically with ties.

Limitations:

- `all_samples` remains an external scoreboard only. Do not use its labels for feature, model, threshold, or calibration selection.
- Existing historical summaries include many all_samples evaluations; if any earlier code selected variants by inspecting all_samples, treat that as a limitation and report candidates as external-scoreboard observations rather than leakage-free selected winners.
- The prediction CLI still cannot generate transition or strict Koopman features for a single text without the required token-loss/hidden-state/lifting artifacts. It now fails clearly instead of silently median-filling most required features.
