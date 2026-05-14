#!/usr/bin/env python3
"""Audit, clean, ablate, tune, and externally evaluate full_allfeatures outputs."""

from __future__ import annotations

import copy
import itertools
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.train_eval import (
    calibration_bins,
    candidate_models,
    class_balance_info,
    detector_metrics,
    get_importance,
    model_ranking_tuple,
    probabilities,
    save_calibration_curve,
    save_plots,
    save_pr_curve,
    save_roc_curve,
)
from src.utils import save_json, write_csv

TRAIN_SETS = ["combined_public", "ghostbuster", "m4", "hc3_plus"]
METADATA_COLUMNS = {
    "id",
    "text",
    "label",
    "source_dataset",
    "language",
    "domain",
    "generator",
    "attack_type",
    "split",
    "type",
    "source",
    "topic",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def group_for_feature(col: str) -> str:
    if col.startswith("burst_"):
        return "burstiness"
    if col.startswith("struct_"):
        return "structure"
    if col.startswith("qwen25_1_5b_"):
        return "probability_qwen25_1_5b"
    if col.startswith("qwen25_7b_"):
        return "probability_qwen25_7b"
    if col.startswith("qwen25_14b_"):
        return "probability_qwen25_14b"
    if col.startswith("qwen25_32b_"):
        return "probability_qwen25_32b"
    if col.startswith("scale_"):
        return "scale_response"
    return "other"


def audit_one(train_set: str) -> tuple[dict, list[dict]]:
    feature_file = ROOT / "features_by_dataset" / f"{train_set}_full_allfeatures" / "all_features.csv"
    checkpoint_dir = ROOT / "checkpoints" / f"{train_set}_full_allfeatures"
    df = pd.read_csv(feature_file)
    feature_columns = read_json(checkpoint_dir / "feature_columns.json", [])
    metadata = read_json(checkpoint_dir / "train_metadata.json", {}) or {}
    numeric_cols = numeric_feature_columns(df)
    used = list(feature_columns)
    present = set(df.columns)
    used_set = set(used)
    unused = [c for c in numeric_cols if c not in used_set]
    missing = [c for c in used if c not in present]
    used_existing = [c for c in used if c in df.columns]
    used_df = df[used_existing].apply(pd.to_numeric, errors="coerce")
    is_inf = np.isinf(used_df.to_numpy(dtype=float, na_value=np.nan))
    nan_counts = used_df.isna().sum()
    inf_counts = pd.Series(is_inf.sum(axis=0), index=used_existing)
    finite_df = used_df.replace([np.inf, -np.inf], np.nan)
    constant_cols = [c for c in used_existing if finite_df[c].dropna().nunique() <= 1]
    cols_32b = [c for c in used_existing if "32b" in c.lower()]
    group_counts = {g: 0 for g in [
        "burstiness",
        "structure",
        "probability_qwen25_1_5b",
        "probability_qwen25_7b",
        "probability_qwen25_14b",
        "probability_qwen25_32b",
        "scale_response",
        "other",
    ]}
    for col in used_existing:
        group_counts[group_for_feature(col)] += 1
    rows_32b = []
    for col in cols_32b:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows_32b.append(
            {
                "train_set": train_set,
                "feature": col,
                "nunique": int(s.nunique(dropna=True)),
                "std": float(s.std()) if s.notna().any() else np.nan,
                "nan_count": int(s.isna().sum()),
                "inf_count": int(np.isinf(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).sum()),
            }
        )
    has_probability = any(c.startswith(("qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_")) for c in used_existing)
    has_scale = any(c.startswith("scale_") for c in used_existing)
    has_32b_probability = any(c.startswith("qwen25_32b_") for c in used_existing)
    has_32b_placeholder = bool(cols_32b) and "qwen25_32b" not in metadata.get("probability_models_used", [])
    audit = {
        "train_set": train_set,
        "all_features_total_columns": int(len(df.columns)),
        "all_features_numeric_feature_columns": int(len(numeric_cols)),
        "training_feature_columns_count": int(len(used)),
        "training_columns_all_exist": bool(not missing),
        "unused_all_features_numeric_columns_count": int(len(unused)),
        "unused_all_features_numeric_columns": ";".join(unused),
        "training_used_missing_columns_count": int(len(missing)),
        "training_used_missing_columns": ";".join(missing),
        "nan_columns_count": int((nan_counts > 0).sum()),
        "inf_columns_count": int((inf_counts > 0).sum()),
        "constant_columns_count": int(len(constant_cols)),
        "constant_columns": ";".join(constant_cols),
        "columns_containing_32b_count": int(len(cols_32b)),
        "columns_containing_32b": ";".join(cols_32b),
        "probability_features_entered_training": bool(has_probability),
        "scale_response_features_entered_training": bool(has_scale),
        "qwen25_32b_probability_entered_training": bool(has_32b_probability),
        "has_32b_placeholder_columns": bool(has_32b_placeholder),
        "drop_recommended_columns_count": int(len(set(constant_cols + cols_32b + missing))),
        "drop_recommended_columns": ";".join(sorted(set(constant_cols + cols_32b + missing))),
        "probability_models_used": ";".join(metadata.get("probability_models_used", [])),
        "scale_response_scales": ";".join(metadata.get("scale_response_scales", []) or metadata.get("available_scales", []) or []),
    }
    audit.update({f"group_count_{k}": v for k, v in group_counts.items()})
    return audit, rows_32b


def run_audit() -> pd.DataFrame:
    rows = []
    rows_32b = []
    for train_set in TRAIN_SETS:
        audit, detail = audit_one(train_set)
        rows.append(audit)
        rows_32b.extend(detail)
    out_dir = ROOT / "results_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(rows)
    write_csv(audit_df, out_dir / "full_allfeatures_feature_audit.csv")
    summary = {
        "created_at": now(),
        "train_sets": TRAIN_SETS,
        "probability_features_entered_training_all": bool(audit_df["probability_features_entered_training"].all()),
        "scale_response_features_entered_training_all": bool(audit_df["scale_response_features_entered_training"].all()),
        "qwen25_32b_probability_entered_training_any": bool(audit_df["qwen25_32b_probability_entered_training"].any()),
        "has_32b_placeholder_columns_any": bool(audit_df["has_32b_placeholder_columns"].any()),
        "columns_containing_32b_by_train_set": dict(zip(audit_df["train_set"], audit_df["columns_containing_32b_count"])),
        "constant_columns_by_train_set": dict(zip(audit_df["train_set"], audit_df["constant_columns_count"])),
        "nan_columns_by_train_set": dict(zip(audit_df["train_set"], audit_df["nan_columns_count"])),
        "inf_columns_by_train_set": dict(zip(audit_df["train_set"], audit_df["inf_columns_count"])),
        "detail_32b": rows_32b,
    }
    save_json(summary, out_dir / "full_allfeatures_feature_audit_summary.json")
    return audit_df


def cleaned_columns(df: pd.DataFrame, metadata: dict, requested: list[str] | None = None) -> tuple[list[str], dict]:
    cols = requested or numeric_feature_columns(df)
    existing = [c for c in cols if c in df.columns and c not in METADATA_COLUMNS]
    numeric = [c for c in existing if pd.api.types.is_numeric_dtype(df[c])]
    work = df[numeric].apply(pd.to_numeric, errors="coerce")
    inf_mask = np.isinf(work.to_numpy(dtype=float, na_value=np.nan))
    inf_counts = pd.Series(inf_mask.sum(axis=0), index=numeric)
    all_nan = work.isna().all()
    all_inf = pd.Series(False, index=numeric)
    for c in numeric:
        arr = work[c].to_numpy(dtype=float, na_value=np.nan)
        all_inf[c] = bool(len(arr) > 0 and np.isinf(arr).all())
    finite = work.replace([np.inf, -np.inf], np.nan)
    constant = finite.nunique(dropna=True) <= 1
    drop = set(all_nan[all_nan].index) | set(all_inf[all_inf].index) | set(constant[constant].index)
    if "qwen25_32b" not in (metadata.get("probability_models_used", []) or []):
        drop |= {c for c in numeric if "32b" in c.lower()}
    kept = [c for c in numeric if c not in drop]
    info = {
        "initial_numeric_columns": int(len(numeric)),
        "dropped_all_nan": sorted(all_nan[all_nan].index.tolist()),
        "dropped_all_inf": sorted(all_inf[all_inf].index.tolist()),
        "dropped_constant": sorted(constant[constant].index.tolist()),
        "dropped_32b": sorted([c for c in numeric if "32b" in c.lower() and c in drop]),
        "columns_with_any_inf": sorted(inf_counts[inf_counts > 0].index.tolist()),
        "final_features": int(len(kept)),
    }
    return kept, info


def make_xy(df: pd.DataFrame, cols: list[str], medians: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    x = df[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        med = x.median(numeric_only=True).fillna(0.0).to_dict()
    else:
        med = medians
    x = x.fillna(med).fillna(0.0)
    y = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return x, y, {k: float(v) for k, v in med.items()}


def split_internal(df: pd.DataFrame, cols: list[str]):
    x, y, med = make_xy(df, cols)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    return (*train_test_split(x, y, df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify), med)


def save_internal_outputs(result_dir: Path, model, model_name: str, metrics_rows: list[dict], x_val, y_val, val_df, cols: list[str], medians: dict, checkpoint_dir: Path, metadata: dict) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prob = probabilities(model, x_val)
    pred = (prob >= 0.5).astype(int)
    best_metrics = detector_metrics(y_val, prob, y_pred=pred, threshold=0.5)
    best_metrics["model"] = model_name
    write_csv(pd.DataFrame(metrics_rows), result_dir / "metrics.csv")
    write_csv(pd.DataFrame([best_metrics]), result_dir / "detector_metrics.csv")
    meta_cols = [c for c in ["id", "text", "label", "type", "source", "topic", "source_dataset", "domain", "generator", "attack_type"] if c in val_df.columns]
    preds = val_df[meta_cols].copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    write_csv(preds, result_dir / "predictions.csv")
    write_csv(save_roc_curve(y_val, prob, result_dir), result_dir / "roc_curve.csv")
    write_csv(save_pr_curve(y_val, prob, result_dir), result_dir / "pr_curve.csv")
    write_csv(save_calibration_curve(y_val, prob, result_dir), result_dir / "calibration_bins.csv")
    importance = get_importance(model, cols)
    write_csv(importance, result_dir / "feature_importance.csv")
    save_plots(y_val, pred, importance, result_dir)
    with open(result_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Best model: {model_name}\n\n")
        handle.write(classification_report(y_val, pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
    joblib.dump(model, checkpoint_dir / "best_model.joblib")
    save_json(cols, checkpoint_dir / "feature_columns.json")
    save_json(medians, checkpoint_dir / "feature_medians.json")
    save_json(metadata, checkpoint_dir / "train_metadata.json")
    return best_metrics


def train_default(df: pd.DataFrame, cols: list[str], result_dir: Path, checkpoint_dir: Path, experiment_name: str, extra_metadata: dict | None = None) -> tuple[object, dict, pd.DataFrame, pd.Series, list[str], dict]:
    x_train, x_val, y_train, y_val, train_df, val_df, medians = split_internal(df, cols)
    rows = []
    best = None
    for name, model in candidate_models(y_train).items():
        model.fit(x_train, y_train)
        prob = probabilities(model, x_val)
        pred = (prob >= 0.5).astype(int)
        metrics = {"model": name}
        metrics.update(detector_metrics(y_val, prob, y_pred=pred, threshold=0.5))
        rows.append(metrics)
        if best is None or model_ranking_tuple(metrics) > model_ranking_tuple(best[0]):
            best = (metrics, model, name)
    best_metrics, best_model, best_name = best
    metadata = {
        "experiment_name": experiment_name,
        "best_model_name": best_name,
        "selection_metric": "auprc>auroc>f1",
        "n_train": int(len(x_train)),
        "n_test_internal": int(len(x_val)),
        "n_features": int(len(cols)),
        "feature_file": "features_by_dataset/combined_public_full_allfeatures/all_features.csv",
        "created_at": now(),
    }
    metadata.update(class_balance_info(y_train))
    if extra_metadata:
        metadata.update(extra_metadata)
    best_metrics = save_internal_outputs(result_dir, best_model, best_name, rows, x_val, y_val, val_df, cols, medians, checkpoint_dir, metadata)
    return best_model, best_metrics, x_val, y_val, cols, medians


def external_frame(feature_file: Path, test_csv: Path, cols: list[str]) -> pd.DataFrame:
    features = pd.read_csv(feature_file)
    test = pd.read_csv(test_csv)
    merged = test.merge(features, on="id", how="left", suffixes=("", "_feature"))
    if "label" not in merged.columns and "label_feature" in merged.columns:
        merged["label"] = merged["label_feature"]
    for col in cols:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged


def evaluate_external(model, checkpoint_dir: Path, output_dir: Path, feature_file: Path, test_csv: Path, threshold: float = 0.5, threshold_name: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = read_json(checkpoint_dir / "feature_columns.json", [])
    medians = read_json(checkpoint_dir / "feature_medians.json", {}) or {}
    merged = external_frame(feature_file, test_csv, cols)
    x, _, _ = make_xy(merged.assign(label=merged["label"].fillna(0)), cols, medians)
    prob = probabilities(model, x)
    pred = (prob >= threshold).astype(int)
    preds = merged.copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    if threshold_name is not None:
        preds["threshold_name"] = threshold_name
        preds["threshold_value"] = threshold
    keep = [c for c in ["id", "text", "label", "ai_probability", "prediction", "threshold_name", "threshold_value", "source_dataset", "domain", "generator", "attack_type"] if c in preds.columns]
    write_csv(preds[keep], output_dir / "predictions.csv")
    valid = merged["label"].notna()
    metrics = {}
    if valid.any():
        y_true = pd.to_numeric(merged.loc[valid, "label"], errors="coerce").astype(int)
        y_prob = prob[valid.to_numpy()]
        y_pred = pred[valid.to_numpy()]
        metrics = detector_metrics(y_true, y_prob, y_pred=y_pred, threshold=threshold)
        write_csv(pd.DataFrame([metrics]), output_dir / "detector_metrics.csv")
        write_csv(pd.DataFrame([metrics]), output_dir / "metrics.csv")
        write_csv(save_roc_curve(y_true, y_prob, output_dir), output_dir / "roc_curve.csv")
        write_csv(save_pr_curve(y_true, y_prob, output_dir), output_dir / "pr_curve.csv")
        write_csv(save_calibration_curve(y_true, y_prob, output_dir), output_dir / "calibration_bins.csv")
        with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
            handle.write(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
    manifest = {
        "created_at": now(),
        "checkpoint_dir": str(checkpoint_dir),
        "test_feature_file": str(feature_file),
        "test_csv": str(test_csv),
        "output_dir": str(output_dir),
        "threshold": threshold,
        "threshold_name": threshold_name,
        "rows": int(len(merged)),
        "has_label": bool(valid.any()),
        "metrics": metrics,
    }
    save_json(manifest, output_dir / "external_eval_manifest.json")
    return metrics


def select_thresholds(y_true, y_prob) -> dict[str, float]:
    precision, recall, thresholds = __import__("sklearn.metrics").metrics.precision_recall_curve(y_true, y_prob)
    if len(thresholds):
        f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best_f1 = float(thresholds[int(np.nanargmax(f1s))])
    else:
        best_f1 = 0.5
    fpr, tpr, roc_thresholds = __import__("sklearn.metrics").metrics.roc_curve(y_true, y_prob)
    out = {"default_threshold": 0.5, "best_f1_threshold": best_f1}
    for target in [0.01, 0.05, 0.10]:
        valid = np.where(fpr <= target + 1e-12)[0]
        if len(valid):
            idx = valid[int(np.argmax(tpr[valid]))]
            val = float(roc_thresholds[idx])
            if not np.isfinite(val):
                val = 1.0
        else:
            val = 1.0
        out[f"target_fpr_{int(target * 100)}pct_threshold"] = val
    return out


def feature_sets(base_cols: list[str]) -> dict[str, list[str]]:
    basic = [c for c in base_cols if c.startswith(("burst_", "struct_"))]
    prob = [c for c in base_cols if c.startswith(("qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_")) and "32b" not in c.lower()]
    scale = [c for c in base_cols if c.startswith("scale_") and "32b" not in c.lower()]
    return {
        "basic_only": basic,
        "probability_only": prob,
        "scale_response_only": scale,
        "basic_plus_probability": basic + prob,
        "basic_plus_scale_response": basic + scale,
        "full_cleaned": basic + prob + scale,
    }


def tuned_candidates(y_train) -> list[tuple[str, object, dict]]:
    candidates = []
    for c in [0.01, 0.1, 1, 10]:
        params = {"C": c, "class_weight": "balanced", "max_iter": 5000}
        candidates.append((
            "logistic_regression",
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(**params))]),
            params,
        ))
    for n, depth, leaf, maxf, cw in itertools.product(
        [300, 500],
        [None, 12, 20],
        [1, 3, 5],
        ["sqrt", "log2"],
        ["balanced", "balanced_subsample"],
    ):
        params = {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": maxf, "class_weight": cw, "n_jobs": -1}
        candidates.append(("random_forest", RandomForestClassifier(random_state=config.RANDOM_STATE, **params), params))
    try:
        from xgboost import XGBClassifier

        balance = class_balance_info(y_train)
        for n, depth, lr in itertools.product([300, 600], [3, 5], [0.05, 0.1]):
            params = {
                "n_estimators": n,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": balance["scale_pos_weight"],
            }
            candidates.append((
                "xgboost",
                XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_STATE, **params),
                params,
            ))
    except Exception as exc:
        warnings.warn(f"xgboost unavailable; skipping XGBoost candidates: {exc}")
    return candidates


def run_tuning(df: pd.DataFrame, cols: list[str]) -> tuple[object, dict]:
    result_dir = ROOT / "results_optimized" / "combined_public_full_allfeatures_tuned"
    checkpoint_dir = ROOT / "checkpoints_optimized" / "combined_public_full_allfeatures_tuned"
    x_train, x_val, y_train, y_val, train_df, val_df, medians = split_internal(df, cols)
    rows = []
    best = None
    for idx, (name, model, params) in enumerate(tuned_candidates(y_train), start=1):
        fitted = copy.deepcopy(model)
        fitted.fit(x_train, y_train)
        prob = probabilities(fitted, x_val)
        pred = (prob >= 0.5).astype(int)
        metrics = {"model": name, "candidate_index": idx, "params_json": json.dumps(params, sort_keys=True)}
        metrics.update(detector_metrics(y_val, prob, y_pred=pred, threshold=0.5))
        rows.append(metrics)
        if best is None or model_ranking_tuple(metrics) > model_ranking_tuple(best[0]):
            best = (metrics, fitted, name, params)
    search_df = pd.DataFrame(rows).sort_values(["auprc", "auroc", "f1"], ascending=False)
    write_csv(search_df, ROOT / "results_optimized" / "hyperparameter_search_results.csv")
    best_metrics, best_model, best_name, best_params = best
    metadata = {
        "experiment_name": "combined_public_full_allfeatures_tuned",
        "best_model_name": best_name,
        "best_params": best_params,
        "selection_metric": "auprc>auroc>f1",
        "n_train": int(len(x_train)),
        "n_test_internal": int(len(x_val)),
        "n_features": int(len(cols)),
        "created_at": now(),
    }
    metadata.update(class_balance_info(y_train))
    save_internal_outputs(result_dir, best_model, best_name, rows, x_val, y_val, val_df, cols, medians, checkpoint_dir, metadata)
    return best_model, {"best_model": best_name, "best_params": best_params, "best_internal_metrics": best_metrics}


def comparison_row(version: str, metrics: dict, n_features: int, best_model: str) -> dict:
    return {
        "model_version": version,
        "n_features": n_features,
        "best_model": best_model,
        "auroc": metrics.get("auroc", np.nan),
        "auprc": metrics.get("auprc", np.nan),
        "f1": metrics.get("f1", np.nan),
        "tpr_at_fpr_1pct": metrics.get("tpr_at_fpr_1pct", np.nan),
        "tpr_at_fpr_5pct": metrics.get("tpr_at_fpr_5pct", np.nan),
        "fpr_at_tpr_95pct": metrics.get("fpr_at_tpr_95pct", np.nan),
        "ece": metrics.get("expected_calibration_error", metrics.get("ECE", np.nan)),
        "brier_score": metrics.get("brier_score", np.nan),
    }


def main() -> None:
    audit_df = run_audit()
    train_feature_file = ROOT / "features_by_dataset" / "combined_public_full_allfeatures" / "all_features.csv"
    external_feature_file = ROOT / "features_external" / "all_samples_full_allfeatures" / "all_features.csv"
    external_test_csv = ROOT / "data" / "test" / "all_samples_prepared.csv"
    metadata = read_json(ROOT / "checkpoints" / "combined_public_full_allfeatures" / "train_metadata.json", {}) or {}
    train_df = pd.read_csv(train_feature_file)
    base_cols, clean_info = cleaned_columns(train_df, metadata)
    clean_info["probability_features_kept"] = int(sum(c.startswith(("qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_")) for c in base_cols))
    clean_info["scale_response_features_kept"] = int(sum(c.startswith("scale_") for c in base_cols))

    clean_model, clean_metrics, x_val, y_val, clean_cols, clean_medians = train_default(
        train_df,
        base_cols,
        ROOT / "results_optimized" / "combined_public_full_allfeatures_cleaned",
        ROOT / "checkpoints_optimized" / "combined_public_full_allfeatures_cleaned",
        "combined_public_full_allfeatures_cleaned",
        {"cleaning_info": clean_info, "probability_models_used": metadata.get("probability_models_used", [])},
    )
    clean_external = evaluate_external(
        clean_model,
        ROOT / "checkpoints_optimized" / "combined_public_full_allfeatures_cleaned",
        ROOT / "results_external" / "combined_public_on_all_samples_full_allfeatures_cleaned",
        external_feature_file,
        external_test_csv,
    )

    ablation_rows = []
    best_ablation = None
    for name, cols in feature_sets(base_cols).items():
        model, internal_metrics, _, _, _, _ = train_default(
            train_df,
            cols,
            ROOT / "results_ablation" / f"combined_public_{name}",
            ROOT / "checkpoints_ablation" / f"combined_public_{name}",
            f"combined_public_{name}",
            {"feature_set": name, "cleaning_info": clean_info},
        )
        external_dir = ROOT / "results_ablation" / f"combined_public_{name}" / "external_eval"
        external_metrics = evaluate_external(
            model,
            ROOT / "checkpoints_ablation" / f"combined_public_{name}",
            external_dir,
            external_feature_file,
            external_test_csv,
        )
        row = comparison_row(name, external_metrics, len(cols), read_json(ROOT / "checkpoints_ablation" / f"combined_public_{name}" / "train_metadata.json", {}).get("best_model_name", ""))
        row["feature_set"] = name
        row["external_test_rows"] = int(len(pd.read_csv(external_test_csv, usecols=["id"])))
        ablation_rows.append(row)
        if best_ablation is None or (row.get("auprc", -np.inf), row.get("auroc", -np.inf), row.get("f1", -np.inf)) > (best_ablation.get("auprc", -np.inf), best_ablation.get("auroc", -np.inf), best_ablation.get("f1", -np.inf)):
            best_ablation = row
    ablation_df = pd.DataFrame(ablation_rows)
    ordered_cols = ["feature_set", "n_features", "best_model", "auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score", "external_test_rows"]
    write_csv(ablation_df[ordered_cols], ROOT / "results_ablation" / "combined_public_feature_ablation_summary.csv")

    val_prob = probabilities(clean_model, x_val)
    thresholds = select_thresholds(y_val, val_prob)
    threshold_rows = []
    threshold_out = ROOT / "results_external" / "combined_public_on_all_samples_full_allfeatures_cleaned_threshold_tuned"
    for threshold_name, threshold in thresholds.items():
        out_dir = threshold_out / threshold_name
        metrics = evaluate_external(
            clean_model,
            ROOT / "checkpoints_optimized" / "combined_public_full_allfeatures_cleaned",
            out_dir,
            external_feature_file,
            external_test_csv,
            threshold=threshold,
            threshold_name=threshold_name,
        )
        threshold_rows.append({
            "threshold_name": threshold_name,
            "threshold_value": threshold,
            "external_accuracy": metrics.get("accuracy", np.nan),
            "external_precision": metrics.get("precision", np.nan),
            "external_recall": metrics.get("recall", np.nan),
            "external_f1": metrics.get("f1", np.nan),
            "external_tpr": metrics.get("tpr", np.nan),
            "external_fpr": metrics.get("fpr", np.nan),
            "external_specificity": metrics.get("specificity", np.nan),
            "external_mcc": metrics.get("mcc", np.nan),
            "external_auroc": metrics.get("auroc", np.nan),
            "external_auprc": metrics.get("auprc", np.nan),
        })
    write_csv(pd.DataFrame(threshold_rows), ROOT / "results_external" / "threshold_tuning_summary.csv")

    tuned_model, tuned_info = run_tuning(train_df, base_cols)
    tuned_external = evaluate_external(
        tuned_model,
        ROOT / "checkpoints_optimized" / "combined_public_full_allfeatures_tuned",
        ROOT / "results_optimized" / "combined_public_full_allfeatures_tuned" / "external_eval",
        external_feature_file,
        external_test_csv,
    )
    tuned_summary = [comparison_row("tuned_full_allfeatures", tuned_external, len(base_cols), tuned_info["best_model"])]
    write_csv(pd.DataFrame(tuned_summary), ROOT / "results_optimized" / "optimized_external_eval_summary.csv")

    original_df = pd.read_csv(ROOT / "results_external" / "external_eval_summary_full_allfeatures.csv")
    original_row = original_df[original_df["train_set"].astype(str).eq("combined_public")]
    if original_row.empty:
        original_metrics = {}
        original_model = ""
    else:
        r = original_row.iloc[0].to_dict()
        original_metrics = {
            "auroc": r.get("auroc", r.get("external_auroc", np.nan)),
            "auprc": r.get("auprc", r.get("external_auprc", np.nan)),
            "f1": r.get("f1", r.get("external_f1", np.nan)),
            "tpr_at_fpr_1pct": r.get("tpr_at_fpr_1pct", np.nan),
            "tpr_at_fpr_5pct": r.get("tpr_at_fpr_5pct", np.nan),
            "fpr_at_tpr_95pct": r.get("fpr_at_tpr_95pct", np.nan),
            "expected_calibration_error": r.get("expected_calibration_error", r.get("ECE", np.nan)),
            "brier_score": r.get("brier_score", np.nan),
        }
        original_model = r.get("best_model", r.get("model", ""))
    comparison = [
        comparison_row("original_combined_public_full_allfeatures", original_metrics, 311, original_model),
        comparison_row("cleaned_full_allfeatures", clean_external, len(base_cols), read_json(ROOT / "checkpoints_optimized" / "combined_public_full_allfeatures_cleaned" / "train_metadata.json", {}).get("best_model_name", "")),
        comparison_row("tuned_full_allfeatures", tuned_external, len(base_cols), tuned_info["best_model"]),
        comparison_row(f"best_ablation_{best_ablation['feature_set']}", best_ablation, int(best_ablation["n_features"]), best_ablation["best_model"]),
    ]
    write_csv(pd.DataFrame(comparison), ROOT / "results_optimized" / "combined_public_cleaned_tuned_comparison.csv")
    save_json(
        {
            "created_at": now(),
            "audit_rows": audit_df.to_dict(orient="records"),
            "cleaning_info": clean_info,
            "cleaned_external_metrics": clean_external,
            "ablation_summary": ablation_rows,
            "thresholds": thresholds,
            "threshold_summary": threshold_rows,
            "tuned_info": tuned_info,
            "tuned_external_metrics": tuned_external,
            "best_ablation": best_ablation,
        },
        ROOT / "results_optimized" / "full_allfeatures_cleanup_optimize_manifest.json",
    )


if __name__ == "__main__":
    main()
