#!/usr/bin/env python3
"""Leakage-safe external scoreboard for all_samples.

This script evaluates already-selected checkpoints on all_samples. It does not
train models, choose thresholds, fit calibration, select features, or choose a
winner from all_samples labels. The only supported threshold here is 0.5.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_eval import calibration_bins, detector_metrics, probabilities


META_COLS = {
    "id",
    "text",
    "label",
    "source_dataset",
    "source",
    "type",
    "language",
    "domain",
    "generator",
    "attack_type",
    "split",
    "topic",
}

FAMILY_PREFIXES = {
    "burst": ("burst_",),
    "struct": ("struct_",),
    "probability": ("qwen25_", "bino_"),
    "scale_response": ("scale_",),
    "transition": ("transition_", "trans_"),
    "strict_koopman": ("strict_koopman_", "text_koopman_"),
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if np.isnan(val) else f"{val:.6g}")
            else:
                vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def clean_name(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep).strip("_")


def merge_on_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if "id" not in left.columns or "id" not in right.columns:
        raise ValueError("Every metadata/feature file must contain an id column.")
    merged = left.merge(right, on="id", how="left", suffixes=("", "_feature"))
    for col in list(merged.columns):
        if col.endswith("_feature"):
            base = col[: -len("_feature")]
            if base not in merged.columns:
                merged[base] = merged[col]
            elif merged[base].isna().all():
                merged[base] = merged[col]
    return merged


def load_matrix(metadata_csv: Path, feature_files: list[Path]) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    for feature_file in feature_files:
        features = pd.read_csv(feature_file)
        df = merge_on_id(df, features)
    if "source_dataset" not in df.columns and "source" in df.columns:
        df["source_dataset"] = df["source"]
    return df


def numeric_feature_candidates(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if col in META_COLS or col.endswith("_feature"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def feature_columns_from_checkpoint(checkpoint_dir: Path, classifier_path: Path, model: Any | None) -> tuple[list[str], str]:
    for candidate in [
        checkpoint_dir / "feature_columns.json",
        classifier_path.with_name("feature_columns.json"),
    ]:
        cols = read_json(candidate, None)
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols], str(candidate)

    if model is not None and hasattr(model, "feature_names_in_"):
        cols = [str(c) for c in model.feature_names_in_]
        if cols:
            return cols, "estimator.feature_names_in_"

    raise ValueError(
        f"Could not determine feature columns for {classifier_path}. "
        "Provide --feature_columns_json or use a checkpoint with feature_columns.json "
        "or a scikit-learn estimator exposing feature_names_in_."
    )


def medians_from_checkpoint(checkpoint_dir: Path, feature_columns: list[str]) -> tuple[dict[str, float], str]:
    for candidate in [checkpoint_dir / "feature_medians.json"]:
        values = read_json(candidate, None)
        if isinstance(values, dict):
            return {str(k): float(v) for k, v in values.items() if isinstance(v, (int, float))}, str(candidate)
    return {col: 0.0 for col in feature_columns}, "zero_fallback_no_training_medians_found"


def feature_family(feature: str) -> str:
    for family, prefixes in FAMILY_PREFIXES.items():
        if feature.startswith(prefixes):
            return family
    return "other"


def align_features(df: pd.DataFrame, feature_columns: list[str], medians: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    aligned = pd.DataFrame(index=df.index)
    for col in feature_columns:
        present = col in df.columns
        series = pd.to_numeric(df[col], errors="coerce") if present else pd.Series(np.nan, index=df.index)
        inf_count = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
        series = series.replace([np.inf, -np.inf], np.nan)
        nan_count = int(series.isna().sum())
        median_value = float(medians.get(col, 0.0))
        aligned[col] = series.fillna(median_value)
        rows.append(
            {
                "feature": col,
                "family": feature_family(col),
                "present": bool(present),
                "missing_column": bool(not present),
                "median_imputed": bool((not present) or nan_count > 0 or inf_count > 0),
                "nan_count_before_impute": nan_count,
                "inf_count_before_impute": inf_count,
                "impute_value": median_value,
            }
        )
    return aligned, pd.DataFrame(rows)


def coverage_summary(report: pd.DataFrame) -> dict[str, Any]:
    required = int(len(report))
    present = int(report["present"].sum()) if required else 0
    missing = required - present
    return {
        "required_features": required,
        "present_features": present,
        "missing_features": missing,
        "missing_feature_pct": float(missing / required) if required else 0.0,
        "median_imputed_columns": int(report["median_imputed"].sum()) if required else 0,
        "nan_count_before_impute": int(report["nan_count_before_impute"].sum()) if required else 0,
        "inf_count_before_impute": int(report["inf_count_before_impute"].sum()) if required else 0,
        "missing_by_family": report.loc[~report["present"], "family"].value_counts().to_dict(),
        "required_by_family": report["family"].value_counts().to_dict(),
    }


def save_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    fpr, tpr, roc_threshold = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_threshold}).to_csv(output_dir / "roc_curve.csv", index=False)
    precision, recall, pr_threshold = precision_recall_curve(y_true, y_prob)
    pr_threshold = np.append(pr_threshold, np.nan)
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": pr_threshold}).to_csv(output_dir / "pr_curve.csv", index=False)
    bins, _ = calibration_bins(y_true, y_prob, n_bins=10)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    major = ["auroc", "auprc", "f1", "mcc", "tpr_at_fpr_5pct", "ECE", "brier_score"]
    if n <= 0:
        return pd.DataFrame(columns=["metric", "mean", "ci_low", "ci_high", "n_bootstrap"])
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in major}
    size = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics = detector_metrics(y_true[idx], y_prob[idx], threshold=0.5)
        for metric in major:
            val = metrics.get(metric)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                values[metric].append(float(val))
    rows = []
    for metric, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(arr)) if arr.size else np.nan,
                "ci_low": float(np.quantile(arr, 0.025)) if arr.size else np.nan,
                "ci_high": float(np.quantile(arr, 0.975)) if arr.size else np.nan,
                "n_bootstrap": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["ai_probability"] = y_prob
    if "text" in work.columns:
        lengths = work["text"].fillna("").astype(str).str.split().str.len()
        try:
            work["length_bucket"] = pd.qcut(lengths.rank(method="first"), q=4, labels=["q1_short", "q2", "q3", "q4_long"])
        except Exception:
            work["length_bucket"] = "all"
    for col in ["source_dataset", "domain", "generator", "attack_type", "length_bucket"]:
        if col not in work.columns:
            continue
        for value, part in work.groupby(col, dropna=False):
            valid = part["label"].notna()
            if valid.sum() < 2 or part.loc[valid, "label"].nunique() < 2:
                continue
            yy = part.loc[valid, "label"].astype(int).to_numpy()
            pp = part.loc[valid, "ai_probability"].to_numpy(dtype=float)
            metrics = detector_metrics(yy, pp, threshold=0.5)
            rows.append({"subgroup_column": col, "subgroup_value": value, "n": int(valid.sum()), **metrics})
    return pd.DataFrame(rows)


def error_analysis(df: pd.DataFrame, y_prob: np.ndarray, output_dir: Path) -> pd.DataFrame:
    work = df.copy()
    work["ai_probability"] = y_prob
    work["prediction"] = (y_prob >= 0.5).astype(int)
    work["abs_margin_from_0_5"] = np.abs(y_prob - 0.5)
    keep = [c for c in ["id", "label", "prediction", "ai_probability", "abs_margin_from_0_5", "source_dataset", "domain", "generator", "attack_type", "text"] if c in work.columns]
    chunks = []
    if "label" in work.columns:
        fp = work[(work["label"].astype(int) == 0) & (work["prediction"] == 1)].sort_values("ai_probability", ascending=False).head(50)
        fn = work[(work["label"].astype(int) == 1) & (work["prediction"] == 0)].sort_values("ai_probability", ascending=True).head(50)
        fp = fp.assign(error_group="high_confidence_false_positive")
        fn = fn.assign(error_group="high_confidence_false_negative")
        chunks.extend([fp, fn])
    near = work.sort_values("abs_margin_from_0_5").head(50).assign(error_group="near_threshold")
    chunks.append(near)

    if "label" in work.columns:
        bins, _ = calibration_bins(work["label"].astype(int).to_numpy(), y_prob, n_bins=10)
        worst_bins = bins.sort_values("abs_calibration_gap", ascending=False).head(3)
        worst_bins.to_csv(output_dir / "worst_calibration_bins.csv", index=False)

    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["error_group", *keep])
    return out[["error_group", *keep]]


def markdown_report(name: str, metrics: dict[str, float], coverage: dict[str, Any], output_dir: Path) -> None:
    lines = [
        f"# all_samples External Scoreboard: {name}",
        "",
        "This report evaluates a preselected checkpoint on all_samples. all_samples labels are not used for training, threshold selection, feature selection, calibration, or model selection by this script.",
        "",
        "## Metrics",
        "",
        f"- AUROC: {metrics.get('auroc', float('nan')):.6f}",
        f"- AUPRC: {metrics.get('auprc', float('nan')):.6f}",
        f"- F1 @ 0.5: {metrics.get('f1', float('nan')):.6f}",
        f"- TPR@FPR=5%: {metrics.get('tpr_at_fpr_5pct', float('nan')):.6f}",
        f"- ECE: {metrics.get('ECE', float('nan')):.6f}",
        f"- Brier: {metrics.get('brier_score', float('nan')):.6f}",
        "",
        "## Feature Coverage",
        "",
        f"- Required features: {coverage['required_features']}",
        f"- Present features: {coverage['present_features']}",
        f"- Missing features: {coverage['missing_features']} ({coverage['missing_feature_pct']:.2%})",
        f"- Median-imputed columns: {coverage['median_imputed_columns']}",
        f"- NaN count before imputation: {coverage['nan_count_before_impute']}",
        f"- Inf count before imputation: {coverage['inf_count_before_impute']}",
    ]
    (output_dir / "external_scoreboard_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def classifier_files(checkpoint_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [checkpoint_dir / name if not Path(name).is_absolute() else Path(name) for name in requested]
    direct = sorted(checkpoint_dir.glob("*classifier.joblib"))
    if (checkpoint_dir / "best_model.joblib").exists():
        direct.insert(0, checkpoint_dir / "best_model.joblib")
    return direct


def evaluate_one(
    checkpoint_dir: Path,
    classifier_path: Path,
    df: pd.DataFrame,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model = joblib.load(classifier_path)
    feature_columns, feature_source = feature_columns_from_checkpoint(checkpoint_dir, classifier_path, model)
    medians, medians_source = medians_from_checkpoint(checkpoint_dir, feature_columns)
    x, coverage_report = align_features(df, feature_columns, medians)
    coverage = coverage_summary(coverage_report)
    y_prob = np.clip(probabilities(model, x), 0.0, 1.0)
    out_name = clean_name(f"{checkpoint_dir.name}__{classifier_path.stem}")
    output_dir = output_root / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = df.copy()
    pred_df["ai_probability"] = y_prob
    pred_df["prediction"] = (y_prob >= 0.5).astype(int)
    keep = [c for c in ["id", "label", "ai_probability", "prediction", "source_dataset", "domain", "generator", "attack_type", "text"] if c in pred_df.columns]
    pred_df[keep].to_csv(output_dir / "predictions.csv", index=False)
    coverage_report.to_csv(output_dir / "feature_coverage_report.csv", index=False)

    metrics: dict[str, float] = {}
    if "label" in df.columns and df["label"].notna().any():
        valid = df["label"].notna().to_numpy()
        y_true = df.loc[valid, "label"].astype(int).to_numpy()
        valid_prob = y_prob[valid]
        metrics = detector_metrics(y_true, valid_prob, threshold=0.5)
        pd.DataFrame([{**metrics, **coverage}]).to_csv(output_dir / "detector_metrics.csv", index=False)
        save_curves(y_true, valid_prob, output_dir)
        bootstrap_ci(y_true, valid_prob, bootstrap_samples, seed).to_csv(output_dir / "bootstrap_ci.csv", index=False)
        subgroup_metrics(df.loc[valid].copy(), valid_prob).to_csv(output_dir / "subgroup_metrics.csv", index=False)
        error_analysis(df.loc[valid].copy(), valid_prob, output_dir).to_csv(output_dir / "error_analysis.csv", index=False)
        markdown_report(out_name, metrics, coverage, output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "leakage_policy": "external_scoreboard_only_no_all_samples_selection_training_thresholding_or_calibration",
        "checkpoint_dir": str(checkpoint_dir),
        "classifier_path": str(classifier_path),
        "feature_column_source": feature_source,
        "median_source": medians_source,
        "rows": int(len(df)),
        "output_dir": str(output_dir),
        "bootstrap_samples_requested": int(bootstrap_samples),
        "coverage": coverage,
        "metrics": metrics,
    }
    write_json(output_dir / "external_scoreboard_manifest.json", manifest)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "classifier": classifier_path.name,
        "output_dir": str(output_dir),
        **coverage,
        **metrics,
    }


def evaluate_existing_predictions(
    predictions_csv: Path,
    scoreboard_name: str,
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    df = pd.read_csv(predictions_csv)
    if "label" not in df.columns or "ai_probability" not in df.columns:
        raise ValueError("--predictions_csv must contain label and ai_probability.")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    y_prob = pd.to_numeric(df["ai_probability"], errors="coerce").clip(0.0, 1.0).to_numpy(dtype=float)
    y_true = df["label"].to_numpy(dtype=int)
    output_dir = output_root / clean_name(scoreboard_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    df["prediction"] = (y_prob >= 0.5).astype(int)
    keep = [c for c in ["id", "label", "ai_probability", "prediction", "source_dataset", "domain", "generator", "attack_type", "text"] if c in df.columns]
    df[keep].to_csv(output_dir / "predictions.csv", index=False)
    metrics = detector_metrics(y_true, y_prob, threshold=0.5)
    coverage = {
        "required_features": np.nan,
        "present_features": np.nan,
        "missing_features": np.nan,
        "missing_feature_pct": np.nan,
        "median_imputed_columns": np.nan,
        "nan_count_before_impute": np.nan,
        "inf_count_before_impute": np.nan,
    }
    pd.DataFrame([{**metrics, **coverage}]).to_csv(output_dir / "detector_metrics.csv", index=False)
    pd.DataFrame(columns=["feature", "family", "present", "missing_column", "median_imputed", "nan_count_before_impute", "inf_count_before_impute", "impute_value"]).to_csv(
        output_dir / "feature_coverage_report.csv", index=False
    )
    save_curves(y_true, y_prob, output_dir)
    bootstrap_ci(y_true, y_prob, bootstrap_samples, seed).to_csv(output_dir / "bootstrap_ci.csv", index=False)
    subgroup_metrics(df.copy(), y_prob).to_csv(output_dir / "subgroup_metrics.csv", index=False)
    error_analysis(df.copy(), y_prob, output_dir).to_csv(output_dir / "error_analysis.csv", index=False)
    markdown_report(scoreboard_name, metrics, {k: (0 if isinstance(v, float) and np.isnan(v) else v) for k, v in coverage.items()}, output_dir)
    write_json(
        output_dir / "external_scoreboard_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "existing_predictions_no_checkpoint_pickle_loaded",
            "leakage_policy": "existing all_samples predictions are evaluated only; no threshold calibration feature or model selection is performed here",
            "predictions_csv": str(predictions_csv),
            "rows": int(len(df)),
            "output_dir": str(output_dir),
            "bootstrap_samples_requested": int(bootstrap_samples),
            "metrics": metrics,
        },
    )
    return {"checkpoint_dir": "existing_predictions", "classifier": scoreboard_name, "output_dir": str(output_dir), **coverage, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dirs", nargs="+")
    parser.add_argument("--classifier_files", nargs="*", default=None, help="Optional classifier filenames to evaluate inside each checkpoint dir.")
    parser.add_argument("--metadata_csv", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--feature_files", nargs="+", default=["features_external/all_samples_full_allfeatures/all_features.csv"])
    parser.add_argument("--predictions_csv", help="Safe curation mode: evaluate an existing predictions.csv without loading a joblib checkpoint.")
    parser.add_argument("--scoreboard_name", default="existing_predictions")
    parser.add_argument("--output_dir", default="results_all_samples_scoreboard")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and write readiness manifest without loading joblib checkpoints.")
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    feature_files = [Path(p) for p in args.feature_files]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_csv:
        row = evaluate_existing_predictions(Path(args.predictions_csv), args.scoreboard_name, output_dir, args.bootstrap_samples, args.seed)
        summary = pd.DataFrame([row])
        summary.to_csv(output_dir / "model_family_comparison.csv", index=False)
        (output_dir / "model_family_comparison.md").write_text(dataframe_to_markdown(summary) + "\n", encoding="utf-8")
        print(f"Wrote predictions-only scoreboard summary to {output_dir / 'model_family_comparison.csv'}")
        return

    if not args.checkpoint_dirs:
        raise ValueError("Provide --checkpoint_dirs for checkpoint evaluation or --predictions_csv for safe curation mode.")

    missing = [str(p) for p in [metadata_csv, *feature_files] if not p.exists()]
    checkpoint_dirs = [Path(p) for p in args.checkpoint_dirs]
    missing.extend(str(p) for p in checkpoint_dirs if not p.exists())
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    df = load_matrix(metadata_csv, feature_files)
    readiness = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_csv": str(metadata_csv),
        "feature_files": [str(p) for p in feature_files],
        "checkpoint_dirs": [str(p) for p in checkpoint_dirs],
        "rows": int(len(df)),
        "has_labels": bool("label" in df.columns and df["label"].notna().any()),
        "numeric_feature_candidates": len(numeric_feature_candidates(df)),
        "dry_run": bool(args.dry_run),
        "leakage_policy": "all_samples is evaluated only after checkpoint/model/feature family selection.",
    }
    write_json(output_dir / "readiness_manifest.json", readiness)
    if args.dry_run:
        print(f"Readiness OK. Wrote {output_dir / 'readiness_manifest.json'}")
        return

    rows = []
    for checkpoint_dir in checkpoint_dirs:
        for classifier_path in classifier_files(checkpoint_dir, args.classifier_files):
            if not classifier_path.exists():
                raise FileNotFoundError(classifier_path)
            rows.append(evaluate_one(checkpoint_dir, classifier_path, df, output_dir, args.bootstrap_samples, args.seed))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        sort_cols = [c for c in ["tpr_at_fpr_5pct", "auroc", "auprc"] if c in summary.columns]
        summary = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    summary.to_csv(output_dir / "model_family_comparison.csv", index=False)
    (output_dir / "model_family_comparison.md").write_text(dataframe_to_markdown(summary) + "\n", encoding="utf-8")
    print(f"Wrote scoreboard summary to {output_dir / 'model_family_comparison.csv'}")


if __name__ == "__main__":
    main()
