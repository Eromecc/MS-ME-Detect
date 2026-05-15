#!/usr/bin/env python3
"""Create non-destructive project inventory, documentation, and curated artifacts."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


KEY_EXTS = {".csv", ".json", ".png", ".pdf", ".joblib", ".parquet", ".gz"}
LARGE_FILE_MB = 25.0

DIR_PURPOSES = {
    "data": "Datasets, train/test splits, strict source splits; private data should not be committed.",
    "features": "Generated feature CSVs for baseline feature extraction.",
    "features_by_dataset": "Per-training-set feature tables, including full_allfeatures.",
    "features_external": "External all_samples feature tables.",
    "features_source_matrix": "Feature subsets extracted for strict source matrix experiments.",
    "features_token_loss": "Token-level loss cache for transition profiling; large generated cache.",
    "features_transition": "Transition-state profiling features generated from token-loss cache.",
    "checkpoints": "Original trained model checkpoints.",
    "checkpoints_ablation": "Ablation model checkpoints.",
    "checkpoints_optimized": "Cleaned/tuned model checkpoints.",
    "checkpoints_source_matrix": "Cross-source generalization model checkpoints.",
    "checkpoints_targeted": "M4-targeted model checkpoints.",
    "results": "Baseline result outputs.",
    "results_by_dataset": "Per-dataset result outputs.",
    "results_external": "External evaluation results and feature audit summaries.",
    "results_optimized": "Cleaned/tuned full_allfeatures results.",
    "results_ablation": "Feature ablation result tables and plots.",
    "results_source_matrix": "Cross-source generalization matrices, shift reports, and plots.",
    "results_diagnosis": "all_samples diagnosis and error analysis plots/tables.",
    "results_targeted": "M4-targeted training result tables and plots.",
    "results_transition": "Transition-state profiling experiments and plots.",
    "results_presentation": "Presentation-ready figures and slide guide.",
    "results_curated": "Curated small tables, manifests, and selected figures for sharing.",
    "project_inventory": "Generated inventory reports.",
    "docs": "Generated project documentation and reproducibility notes.",
    "src": "Reusable project source code.",
    "scripts": "Experiment and utility scripts.",
    "logs": "Run logs and generated execution output.",
    "__pycache__": "Python bytecode cache; safe to delete after listing.",
}

COMMIT_RECOMMENDATIONS = {
    ".py": True,
    ".md": True,
    ".txt": True,
    ".yml": True,
    ".yaml": True,
    ".toml": True,
    ".json": True,
    ".csv": "small",
    ".png": "selected",
    ".pdf": "selected",
    ".joblib": False,
    ".parquet": False,
    ".gz": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def sizeof(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def dir_stats(path: Path) -> dict:
    files = 0
    dirs = 0
    size = 0
    latest = 0.0
    for base, dirnames, filenames in os.walk(path):
        dirs += len(dirnames)
        files += len(filenames)
        for name in filenames:
            p = Path(base) / name
            try:
                st = p.stat()
            except OSError:
                continue
            size += st.st_size
            latest = max(latest, st.st_mtime)
    return {
        "file_count": files,
        "subdir_count": dirs,
        "size_mb": round(size / (1024 * 1024), 3),
        "last_modified": datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest else "",
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_purpose(path: Path) -> str:
    parts = path.parts
    name = path.name
    top = parts[0] if parts else name
    if top in DIR_PURPOSES:
        return DIR_PURPOSES[top]
    if name.endswith("_summary.csv") or name.endswith("_summary.json"):
        return "Experiment summary table or manifest."
    if name.endswith("predictions.csv"):
        return "Per-sample model predictions."
    if "roc_curve" in name or "pr_curve" in name or "calibration" in name:
        return "Evaluation curve or calibration output."
    if name.endswith(".joblib"):
        return "Serialized model checkpoint."
    if name.endswith(".jsonl.gz"):
        return "Compressed generated cache, likely token-level sequence data."
    return "Project artifact."


def file_type(path: Path) -> str:
    if path.name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    return path.suffix.lstrip(".") or "unknown"


def can_commit(path: Path, size_mb: float) -> bool:
    s = path.as_posix()
    ext = ".jsonl.gz" if s.endswith(".jsonl.gz") else path.suffix
    rec = COMMIT_RECOMMENDATIONS.get(ext)
    if rec is False:
        return False
    if rec == "small":
        return size_mb <= 1.0 and not s.startswith("data/")
    if rec == "selected":
        return size_mb <= 5.0 and ("results_presentation" in s or "results_curated" in s)
    if s.startswith(("data/", "features_", "checkpoints", "results_transition")):
        return False
    return bool(rec)


def recommended_action(path: Path, size_mb: float) -> str:
    s = path.as_posix()
    if path.name.endswith(".joblib") or s.startswith("checkpoints"):
        return "do_not_commit"
    if path.name.endswith(".jsonl.gz") or "features_token_loss" in s:
        return "archive"
    if s.startswith("data/") or "features_by_dataset" in s or "features_external" in s:
        return "ignore"
    if size_mb >= LARGE_FILE_MB:
        return "archive"
    return "keep"


def inventory(root: Path, inventory_dir: Path) -> dict:
    root_rows = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name == ".git":
            continue
        if entry.is_dir():
            st = dir_stats(entry)
            root_rows.append({
                "path": entry.name,
                **st,
                "likely_purpose": DIR_PURPOSES.get(entry.name, "Unclassified top-level directory."),
            })
        else:
            size_mb = sizeof(entry) / (1024 * 1024)
            root_rows.append({
                "path": entry.name,
                "file_count": 1,
                "subdir_count": 0,
                "size_mb": round(size_mb, 3),
                "last_modified": datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc).isoformat(),
                "likely_purpose": "Top-level project file.",
            })
    write_csv(inventory_dir / "root_directory_inventory.csv", root_rows)

    key_rows = []
    large_rows = []
    artifact_map: dict[str, list[dict]] = {}
    for base, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            p = Path(base) / name
            rp = Path(rel(p, root))
            ext = ".jsonl.gz" if name.endswith(".jsonl.gz") else p.suffix
            size_mb = sizeof(p) / (1024 * 1024)
            if ext in KEY_EXTS or ext == ".jsonl.gz":
                row = {
                    "path": rp.as_posix(),
                    "size_mb": round(size_mb, 3),
                    "file_type": file_type(p),
                    "last_modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                    "likely_purpose": infer_purpose(rp),
                }
                key_rows.append(row)
                artifact_map.setdefault(rp.parts[0], []).append(row)
            if size_mb >= LARGE_FILE_MB or ext in {".joblib", ".parquet", ".gz", ".jsonl.gz"}:
                large_rows.append({
                    "path": rp.as_posix(),
                    "size_mb": round(size_mb, 3),
                    "file_type": file_type(p),
                    "likely_purpose": infer_purpose(rp),
                    "can_commit_to_git": str(can_commit(rp, size_mb)).lower(),
                    "recommended_action": recommended_action(rp, size_mb),
                })
    write_csv(inventory_dir / "key_artifact_inventory.csv", key_rows)
    write_csv(inventory_dir / "large_files_report.csv", sorted(large_rows, key=lambda r: r["size_mb"], reverse=True))
    (inventory_dir / "generated_artifact_map.json").write_text(json.dumps(artifact_map, indent=2), encoding="utf-8")
    return {"root_rows": root_rows, "key_rows": key_rows, "large_rows": large_rows}


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fmt_float(x, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_Not available._\n"
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "_No requested columns found._\n"
    sub = df[cols].head(max_rows).copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda v: fmt_float(v))
    return df_to_markdown(sub) + "\n"


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional tabulate dependency."""
    if df is None or df.empty:
        return "_Not available._"
    headers = [str(c) for c in df.columns]
    body = []
    for _, row in df.iterrows():
        body.append([("" if pd.isna(v) else str(v)) for v in row.tolist()])
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"
    lines = [fmt_row(headers), "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"]
    lines.extend(fmt_row(row) for row in body)
    return "\n".join(lines)


def best_row(path: Path, metric: str = "auroc") -> pd.Series | None:
    df = read_csv_if_exists(path)
    if df is None or df.empty or metric not in df.columns:
        return None
    return df.sort_values(metric, ascending=False).iloc[0]


def docs_project_index(docs: Path, root_rows: list[dict]) -> None:
    rows = []
    for name in [
        "data", "data/raw", "data/train_sets", "data/source_splits", "features", "features_by_dataset",
        "features_external", "features_source_matrix", "features_token_loss", "features_transition",
        "checkpoints", "checkpoints_ablation", "checkpoints_optimized", "checkpoints_source_matrix",
        "checkpoints_targeted", "results", "results_by_dataset", "results_external", "results_optimized",
        "results_ablation", "results_source_matrix", "results_diagnosis", "results_targeted",
        "results_transition", "results_presentation", "src", "scripts", "logs",
    ]:
        top = name.split("/")[0]
        rows.append({"Directory": name + "/", "Purpose": DIR_PURPOSES.get(name, DIR_PURPOSES.get(top, "Project directory."))})
    content = f"""# Project Index

Generated: {now()}

## Project Purpose

This project studies LLM-generated text detection. Current experiments focus on probability features, scale-response profiling, cross-source generalization diagnosis, and transition-state profiling from token-level loss sequences.

## Directory Guide

{df_to_markdown(pd.DataFrame(rows))}

## What Should Be Committed To GitHub

Recommended:
- `src/`
- `scripts/`
- `docs/`
- README and small configuration files
- small CSV summaries that document headline results
- selected presentation figures from `results_curated/figures/`

Recommended not to commit:
- raw datasets and private test data
- local model files
- token-loss caches
- large feature CSVs
- checkpoint `.joblib` files
- large plot archives
- `__pycache__/` and `*.pyc`

## Current Best Result

Current best external `all_samples` setup:

`leave_out_ghostbuster + full_plus_1_5b_and_7b_transition`

| Metric | Value |
|---|---:|
| AUROC | 0.6951 |
| AUPRC | 0.6592 |
| F1 | 0.6799 |
| TPR@FPR=5% | 0.0933 |
| ECE | 0.1488 |
| Brier | 0.2459 |

## Main Conclusion

Public benchmark in-domain performance is very high, but `all_samples` is a strongly shifted external set. Ghostbuster-trained models show probability reversal on `all_samples`. Scale-response improves external ranking, and full-scale transition-state profiling further improves the best `all_samples` setup. Low-FPR detection remains weak and needs more calibration or training data alignment.
"""
    (docs / "PROJECT_INDEX.md").write_text(content, encoding="utf-8")


def docs_results_summary(root: Path, docs: Path) -> None:
    sections = ["# Results Summary\n", f"Generated: {now()}\n"]
    specs = [
        ("Basic Baseline", "results_external/external_eval_summary_basic.csv", "Initial external baseline.", ["train_set", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece"]),
        ("Full Allfeatures", "results_external/external_eval_summary_full_allfeatures.csv", "Original full feature set with probability and scale-response summaries.", ["train_set", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece"]),
        ("Cleaned / Tuned", "results_optimized/combined_public_cleaned_tuned_comparison.csv", "Cleaned full_allfeatures and tuned comparison.", ["model_version", "n_features", "best_model", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece", "brier_score"]),
        ("Optimized External Eval", "results_optimized/optimized_external_eval_summary.csv", "Optimized model external evaluation.", ["model_version", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece", "brier_score"]),
        ("Source Generalization", "results_source_matrix/source_generalization_matrix.csv", "Cross-source train/test matrix.", ["train_name", "test_name", "best_model", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece"]),
        ("M4 Targeted", "results_targeted/m4_targeted_summary.csv", "M4-targeted training variants for all_samples.", ["train_variant", "best_model", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece", "brier_score"]),
        ("All Samples Model Comparison", "results_targeted/all_samples_model_comparison.csv", "Comparison of prior and targeted all_samples models.", ["model_version", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece", "brier_score"]),
        ("Transition 1.5B Optimized", "results_transition/fullscale_1_5b_optimized/transition_optimized_summary.csv", "Full-scale 1.5B transition feature selection and model comparison.", ["train_name", "experiment", "test_set", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece"]),
        ("Transition 1.5B Late Fusion", "results_transition/fullscale_1_5b_optimized/late_fusion_summary.csv", "Late fusion of full and transition-only scores.", ["train_name", "test_set", "alpha", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece"]),
        ("Transition 1.5B Feature Selection", "results_transition/fullscale_1_5b_optimized/feature_selection_summary.csv", "Transition feature selection variants.", ["train_name", "feature_set", "test_set", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece"]),
        ("Transition 7B Targeted", "results_transition/qwen25_7b_targeted/transition_7b_summary.csv", "Targeted 7B transition validation against 1.5B and dual-scale features.", ["train_name", "experiment", "test_set", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece", "brier_score"]),
    ]
    for title, path_str, purpose, cols in specs:
        path = root / path_str
        df = read_csv_if_exists(path)
        sections.append(f"\n## {title}\n\nPurpose: {purpose}\n\nSource: `{path_str}`\n\n")
        if df is None:
            sections.append("_Missing or unreadable; skipped._\n")
            continue
        if "test_set" in df.columns and "all_samples" in set(df["test_set"].astype(str)):
            view = df[df["test_set"].astype(str).eq("all_samples")]
            if "auroc" in view.columns:
                view = view.sort_values("auroc", ascending=False)
        elif "auroc" in df.columns:
            view = df.sort_values("auroc", ascending=False)
        else:
            view = df
        sections.append(markdown_table(view, cols, max_rows=10))
        sections.append("\nRecommendation: continue only where external `all_samples` improves without using it for tuning. Limitation: external metrics remain sensitive to distribution shift and low-FPR operating points.\n")
    manifest = root / "results_transition/qwen25_7b_targeted/transition_7b_manifest.json"
    sections.append("\n## Transition 7B Probe Manifest\n\n")
    if manifest.exists():
        d = json.loads(manifest.read_text())
        sections.append("```json\n" + json.dumps(d.get("pca_probe", {}), indent=2) + "\n```\n")
    else:
        sections.append("_Missing._\n")
    (docs / "RESULTS_SUMMARY.md").write_text("\n".join(sections), encoding="utf-8")


def docs_repro(docs: Path) -> None:
    stages = [
        ("Dataset preparation", "python scripts/build_source_splits.py --input data/dataset_english_v1.csv --output_dir data/source_splits --seed 42", "data/source_splits/", "Low", "No", "Safe; overwrites generated split CSVs only."),
        ("Basic baseline", "python main.py --features basic", "results/, results_external/", "Medium", "No", "Use care; may overwrite baseline outputs depending on main.py args."),
        ("Full allfeatures", "python main.py --features full_allfeatures", "features_by_dataset/, checkpoints/, results_external/", "High", "Yes for probability/scale-response if not cached", "Not recommended unless caches and output paths are controlled."),
        ("Cleanup / ablation / tuning", "python scripts/run_full_allfeatures_optimization.py", "results_optimized/, checkpoints_optimized/, results_ablation/", "Medium", "No", "Safe if output dirs are optimized/ablation only."),
        ("Source matrix", "python scripts/run_source_matrix_eval.py --features features_by_dataset/combined_public_full_allfeatures/all_features.csv --external_features features_external/all_samples_full_allfeatures/all_features.csv --source_splits data/source_splits --external_test data/test/all_samples_prepared.csv --output_dir results_source_matrix --checkpoint_dir checkpoints_source_matrix --seed 42", "results_source_matrix/, checkpoints_source_matrix/", "Medium", "No", "Safe; uses existing features."),
        ("Diagnosis / targeted", "python scripts/run_m4_targeted_diagnosis.py", "results_diagnosis/, results_targeted/", "Medium", "No", "Safe if output dirs are diagnosis/targeted only."),
        ("Presentation figures", "python scripts/make_presentation_figures.py --output_dir results_presentation --dpi 300", "results_presentation/", "Low", "No", "Safe; redraws presentation copies."),
        ("Transition smoke", "python scripts/run_transition_profile_experiment.py", "features_transition/, results_transition/", "Low to Medium", "Only if token loss cache missing", "Use max_rows for smoke tests."),
        ("Transition fullscale 1.5B", "python scripts/run_transition_fullscale_optimized.py --model_name qwen25_1_5b --resume --seed 42", "results_transition/fullscale_1_5b_optimized/", "High", "Yes if cache missing", "Safe with resume and fixed output dir; avoid overwriting completed results."),
        ("Transition targeted 7B", "python scripts/run_transition_7b_targeted.py --model_name qwen25_7b --train_sources m4 combined_strict leave_out_ghostbuster --test_sets all_samples m4_test ghostbuster_test hc3_plus_test --resume --seed 42 --max_length 256", "results_transition/qwen25_7b_targeted/", "High", "Yes if cache missing", "Safe with resume; does not run 14B."),
    ]
    rows = [{"Stage": a, "Command": f"`{b}`", "Expected outputs": c, "Estimated cost": d, "Large model inference": e, "Safe to rerun": f} for a, b, c, d, e, f in stages]
    content = "# Reproducibility Commands\n\nThese commands document prior experiment stages. They are not executed by the organization script.\n\n"
    content += df_to_markdown(pd.DataFrame(rows)) + "\n"
    (docs / "REPRODUCIBILITY_COMMANDS.md").write_text(content, encoding="utf-8")


def docs_transition_summary(docs: Path) -> None:
    content = """# Transition-State Profiling Summary

## Motivation

Transition-state profiling tests a content/structure separation idea: instead of using raw token IDs or token strings, token-level losses are mapped into abstract states and summarized as transition patterns.

## Current Implementation

- token-level loss cache saved as compressed JSONL
- no raw token strings are stored
- loss quantile states
- 3/5/7-state transition matrices
- transition entropy, self-transition rate, upward/downward rates
- high-loss burst and low/high run-length features
- spectral gap
- train-only bins for state thresholds to avoid `all_samples` leakage

## 1.5B Fullscale Result

Best 1.5B setup:

`leave_out_ghostbuster + full_plus_transition`

| Metric | Value |
|---|---:|
| AUROC | 0.6816 |
| AUPRC | 0.6480 |
| F1 | 0.6833 |
| TPR@FPR=5% | 0.0733 |
| ECE | 0.1370 |

## 7B Targeted Result

Best targeted 7B setup:

`leave_out_ghostbuster + full_plus_1_5b_and_7b_transition`

| Metric | Value |
|---|---:|
| AUROC | 0.6951 |
| AUPRC | 0.6592 |
| F1 | 0.6799 |
| TPR@FPR=5% | 0.0933 |
| ECE | 0.1488 |
| Brier | 0.2459 |

## Interpretation

The transition signal is real: it improves the best external ranking setup and label probe accuracy remains higher than source/domain probes. Dual-scale 1.5B+7B transition features provide incremental value, while `leave_out_ghostbuster` remains the best training source for `all_samples`. Low-FPR detection remains weak, so feature selection, calibration, and source-aware validation should be prioritized before expensive 14B transition runs.

## Relation To Koopman-Inspired Idea

This is Koopman-inspired transition-state profiling, not a full Deep DMD or Koopman operator implementation. The current method discretizes loss dynamics into abstract states and summarizes transition behavior; it does not learn a global Koopman operator in latent space.
"""
    (docs / "TRANSITION_STATE_PROFILING_SUMMARY.md").write_text(content, encoding="utf-8")


def docs_gitignore(root: Path, docs: Path, inventory_dir: Path) -> None:
    suggestions = [
        "__pycache__/", "*.pyc", "data/raw/", "models/", "features_token_loss/",
        "features_transition/formal/", "features_by_dataset/", "features_external/",
        "features_source_matrix/", "checkpoints*/", "results*/**/*.joblib", "logs/",
        "*.jsonl.gz", "*.parquet", "*.npz",
    ]
    content = "# Gitignore Suggestions\n\nDo not overwrite `.gitignore` automatically. Suggested ignore rules:\n\n"
    content += "\n".join(f"- `{s}`" for s in suggestions) + "\n"
    (docs / "GITIGNORE_SUGGESTIONS.md").write_text(content, encoding="utf-8")
    existing = set()
    gitignore = root / ".gitignore"
    if gitignore.exists():
        existing = {line.strip() for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.strip().startswith("#")}
    gaps = [s for s in suggestions if s not in existing]
    report = ["# .gitignore Gap Report", "", "Existing `.gitignore` was read but not modified.", "", "Missing suggested patterns:"]
    report += [f"- {s}" for s in gaps] if gaps else ["- None"]
    (inventory_dir / "gitignore_gap_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def docs_cleanup_plan(root: Path, docs: Path) -> None:
    pycache = [rel(p, root) for p in root.rglob("__pycache__") if ".git" not in p.parts]
    content = "# Cleanup Plan\n\nNo files were deleted by this organization run.\n\n"
    content += "## Cache Paths That Are Usually Safe To Delete After Listing\n\n"
    content += "\n".join(f"- `{p}/`" for p in pycache) if pycache else "- No `__pycache__` directories found."
    content += "\n\n## Do Not Delete Without Archival\n\n"
    content += "- `features_token_loss/`: expensive token-level cache\n- `features_by_dataset/`: generated feature tables used by downstream scripts\n- `features_external/`: external feature tables\n- `checkpoints*/`: trained model checkpoints\n- `results_transition/`: transition experiment records and plots\n"
    content += "\n## Can Be Archived\n\n- Older result directories after final summaries are copied to `results_curated/`\n- Large plot archives not needed for presentation\n- Intermediate predictions where summary tables already exist\n"
    content += "\n## Do Not Move In Place\n\n- `src/`, `scripts/`, `data/`, and generated feature directories are path dependencies for existing scripts.\n"
    (docs / "CLEANUP_PLAN.md").write_text(content, encoding="utf-8")


def unique_destination(src: Path, dst_dir: Path, root: Path) -> Path:
    dst = dst_dir / src.name
    if not dst.exists():
        return dst
    rp = rel(src, root).replace("/", "__")
    return dst_dir / rp


def copy_if_exists(src: Path, dst_dir: Path, copied: list[str], skipped: list[str], root: Path) -> None:
    if not src.exists():
        skipped.append(rel(src, root))
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = unique_destination(src, dst_dir, root)
    shutil.copy2(src, dst)
    copied.append(rel(dst, root))


def copy_glob(root: Path, pattern: str, dst_dir: Path, copied: list[str], skipped: list[str]) -> None:
    matches = sorted(root.glob(pattern))
    if not matches:
        skipped.append(pattern)
        return
    for src in matches:
        if src.is_file():
            copy_if_exists(src, dst_dir, copied, skipped, root)


def curate(root: Path, curated: Path) -> dict:
    copied: list[str] = []
    skipped: list[str] = []
    table_files = [
        "results_external/external_eval_summary_basic.csv",
        "results_external/external_eval_summary_full_allfeatures.csv",
        "results_external/external_eval_summary_basic_vs_full_allfeatures.csv",
        "results_optimized/combined_public_cleaned_tuned_comparison.csv",
        "results_ablation/combined_public_feature_ablation_summary.csv",
        "results_source_matrix/source_generalization_matrix.csv",
        "results_source_matrix/source_generalization_matrix_with_ci.csv",
        "results_source_matrix/distribution_shift_report.csv",
        "results_targeted/m4_targeted_summary.csv",
        "results_targeted/all_samples_model_comparison.csv",
        "results_transition/fullscale_1_5b_optimized/transition_optimized_summary.csv",
        "results_transition/fullscale_1_5b_optimized/late_fusion_summary.csv",
        "results_transition/fullscale_1_5b_optimized/feature_selection_summary.csv",
        "results_transition/qwen25_7b_targeted/transition_7b_summary.csv",
        "results_transition/qwen25_7b_targeted/transition_7b_vs_1_5b_comparison.csv",
    ]
    for f in table_files:
        copy_if_exists(root / f, curated / "tables", copied, skipped, root)
    manifest_files = [
        "results_transition/qwen25_7b_targeted/transition_7b_manifest.json",
        "results_external/full_allfeatures_feature_audit_summary.json",
        "checkpoints_optimized/combined_public_full_allfeatures_cleaned/train_metadata.json",
        "checkpoints_optimized/combined_public_full_allfeatures_tuned/train_metadata.json",
    ]
    for f in manifest_files:
        copy_if_exists(root / f, curated / "manifests", copied, skipped, root)
    for pattern in ["checkpoints_source_matrix/*/train_metadata.json", "checkpoints_targeted/*/train_metadata.json"]:
        copy_glob(root, pattern, curated / "best_model_metadata", copied, skipped)
    if (root / "results_presentation").exists():
        copy_glob(root, "results_presentation/figures_clean/*.png", curated / "figures", copied, skipped)
        copy_glob(root, "results_presentation/figures_clean/*.pdf", curated / "figures", copied, skipped)
        copy_if_exists(root / "results_presentation/FIGURE_INDEX.md", curated / "figures", copied, skipped, root)
        copy_if_exists(root / "results_presentation/SLIDE_GUIDE.md", curated / "figures", copied, skipped, root)
    else:
        skipped.append("results_presentation/")
    (curated / "CURATION_MANIFEST.json").write_text(json.dumps({"created_at": now(), "copied": copied, "skipped": skipped}, indent=2), encoding="utf-8")
    return {"copied": copied, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_docs", default="docs")
    parser.add_argument("--curated_dir", default="results_curated")
    parser.add_argument("--inventory_dir", default="project_inventory")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    docs = root / args.output_docs
    curated = root / args.curated_dir
    inventory_dir = root / args.inventory_dir
    docs.mkdir(parents=True, exist_ok=True)
    curated.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)

    inv = inventory(root, inventory_dir)
    docs_project_index(docs, inv["root_rows"])
    docs_results_summary(root, docs)
    docs_repro(docs)
    docs_transition_summary(docs)
    docs_gitignore(root, docs, inventory_dir)
    docs_cleanup_plan(root, docs)
    cur = curate(root, curated)
    # Refresh inventory after generated docs and curated copies exist.
    inv = inventory(root, inventory_dir)

    print(json.dumps({
        "created_at": now(),
        "docs": sorted(p.name for p in docs.glob("*.md")),
        "inventory_files": sorted(p.name for p in inventory_dir.iterdir() if p.is_file()),
        "curated_copied_count": len(cur["copied"]),
        "curated_skipped_count": len(cur["skipped"]),
        "deleted_files": 0,
        "modified_gitignore": False,
    }, indent=2))


if __name__ == "__main__":
    main()
