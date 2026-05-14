#!/usr/bin/env python3
"""Run source-specific cross-source generalization diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.train_eval import (
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

SOURCES = ["ghostbuster", "m4", "hc3_plus"]
SPLITS = ["train", "dev", "test"]
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


def select_clean_columns(feature_df: pd.DataFrame) -> tuple[list[str], dict]:
    numeric = [c for c in feature_df.columns if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[c])]
    allowed = [
        c
        for c in numeric
        if c.startswith(("burst_", "struct_", "scale_", "qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_"))
        and "32b" not in c.lower()
    ]
    work = feature_df[allowed].apply(pd.to_numeric, errors="coerce")
    inf = np.isinf(work.to_numpy(dtype=float, na_value=np.nan))
    inf_counts = pd.Series(inf.sum(axis=0), index=allowed)
    finite = work.replace([np.inf, -np.inf], np.nan)
    all_nan = finite.isna().all()
    constant = finite.nunique(dropna=True) <= 1
    drop = set(all_nan[all_nan].index) | set(inf_counts[inf_counts > 0].index) | set(constant[constant].index)
    cols = [c for c in allowed if c not in drop]
    info = {
        "initial_numeric_columns": len(numeric),
        "candidate_cleaned_full_allfeatures": len(allowed),
        "final_features": len(cols),
        "dropped_all_nan": sorted(all_nan[all_nan].index.tolist()),
        "dropped_inf": sorted(inf_counts[inf_counts > 0].index.tolist()),
        "dropped_constant": sorted(constant[constant].index.tolist()),
        "dropped_32b_count": int(sum("32b" in c.lower() for c in numeric)),
        "n_burstiness": int(sum(c.startswith("burst_") for c in cols)),
        "n_structure": int(sum(c.startswith("struct_") for c in cols)),
        "n_probability": int(sum(c.startswith(("qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_")) for c in cols)),
        "n_scale_response": int(sum(c.startswith("scale_") for c in cols)),
    }
    return cols, info


def load_split(source_splits: Path, source: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(source_splits / f"{source}_strict_{split}.csv")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return df


def extract_feature_splits(features_path: Path, source_splits: Path, clean_cols: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    feature_df = pd.read_csv(features_path)
    feature_by_id = feature_df[["id", *clean_cols]].copy()
    out_root = ROOT / "features_source_matrix"
    out_root.mkdir(parents=True, exist_ok=True)
    split_frames = {}
    missing_rows = []
    summary_rows = []
    for source in SOURCES:
        for split in SPLITS:
            meta = load_split(source_splits, source, split)
            merged = meta.merge(feature_by_id, on="id", how="left", validate="one_to_one")
            missing = merged[clean_cols].isna().all(axis=1)
            for missing_id in merged.loc[missing, "id"].astype(str).tolist():
                missing_rows.append({"source_dataset": source, "split": split, "id": missing_id})
            split_key = f"{source}_{split}"
            split_dir = out_root / split_key
            split_dir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(split_dir / "all_features.csv", index=False)
            split_frames[split_key] = merged
            summary_rows.append(
                {
                    "split_key": split_key,
                    "source_dataset": source,
                    "split": split,
                    "rows": int(len(merged)),
                    "missing_ids": int(missing.sum()),
                    "n_features": int(len(clean_cols)),
                }
            )
    missing_df = pd.DataFrame(missing_rows, columns=["source_dataset", "split", "id"])
    missing_df.to_csv(out_root / "missing_ids.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_root / "feature_extraction_summary.csv", index=False)
    return split_frames, missing_df


def make_xy(df: pd.DataFrame, cols: list[str], medians: dict[str, float] | None = None):
    x = df[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(medians).fillna(0.0)
    y = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return x, y, {k: float(v) for k, v in medians.items()}


def candidate_models(y_train: pd.Series):
    models = [
        ("logistic_regression", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))])),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
    ]
    try:
        from xgboost import XGBClassifier

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=config.RANDOM_STATE,
                    scale_pos_weight=neg / max(pos, 1),
                ),
            )
        )
    except Exception as exc:
        warnings.warn(f"xgboost unavailable; skipping XGBoost candidates: {exc}")
    return models


def train_one(train_name: str, train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str], checkpoint_root: Path):
    x_train, y_train, medians = make_xy(train_df, cols)
    x_dev, y_dev, _ = make_xy(dev_df, cols, medians)
    rows = []
    best = None
    for model_name, model in candidate_models(y_train):
        fitted = copy.deepcopy(model)
        fitted.fit(x_train, y_train)
        prob = probabilities(fitted, x_dev)
        pred = (prob >= 0.5).astype(int)
        metrics = {"model": model_name}
        metrics.update(detector_metrics(y_dev, prob, y_pred=pred, threshold=0.5))
        rows.append(metrics)
        if best is None or model_ranking_tuple(metrics) > model_ranking_tuple(best[0]):
            best = (metrics, fitted, model_name)
    best_metrics, best_model, best_name = best
    ckpt = checkpoint_root / train_name
    ckpt.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ckpt / "best_model.joblib")
    save_json(cols, ckpt / "feature_columns.json")
    save_json(medians, ckpt / "feature_medians.json")
    metadata = {
        "created_at": now(),
        "train_name": train_name,
        "best_model_name": best_name,
        "selection_metric": "auprc>auroc>f1",
        "n_train": int(len(train_df)),
        "n_dev": int(len(dev_df)),
        "n_features": int(len(cols)),
        "dev_metrics": best_metrics,
        "candidate_dev_metrics": rows,
    }
    save_json(metadata, ckpt / "train_metadata.json")
    return best_model, best_name, medians, metadata


def save_eval_outputs(model, train_name: str, test_name: str, test_df: pd.DataFrame, cols: list[str], medians: dict, out_dir: Path, best_model: str, n_train: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, _ = make_xy(test_df, cols, medians)
    prob = probabilities(model, x)
    pred = (prob >= 0.5).astype(int)
    metrics = detector_metrics(y, prob, y_pred=pred, threshold=0.5)
    write_csv(pd.DataFrame([metrics]), out_dir / "metrics.csv")
    write_csv(pd.DataFrame([metrics]), out_dir / "detector_metrics.csv")
    keep = [c for c in ["id", "text", "label", "source_dataset", "domain", "generator", "attack_type", "split"] if c in test_df.columns]
    preds = test_df[keep].copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    write_csv(preds, out_dir / "predictions.csv")
    write_csv(save_roc_curve(y, prob, out_dir), out_dir / "roc_curve.csv")
    write_csv(save_pr_curve(y, prob, out_dir), out_dir / "pr_curve.csv")
    write_csv(save_calibration_curve(y, prob, out_dir), out_dir / "calibration_bins.csv")
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(classification_report(y, pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
    manifest = {
        "created_at": now(),
        "train_name": train_name,
        "test_name": test_name,
        "n_train": n_train,
        "n_test": int(len(test_df)),
        "n_features": int(len(cols)),
        "best_model": best_model,
        "metrics": metrics,
    }
    save_json(manifest, out_dir / "eval_manifest.json")
    row = {
        "train_name": train_name,
        "test_name": test_name,
        "n_train": n_train,
        "n_test": int(len(test_df)),
        "n_features": int(len(cols)),
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
    np.savez_compressed(out_dir / "bootstrap_arrays.npz", y=y.to_numpy(dtype=int), prob=prob, pred=pred)
    return row


def balanced_source_label_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups = list(df.groupby(["source_dataset", "label"], dropna=False))
    min_n = min(len(g) for _, g in groups)
    return pd.concat([g.sample(n=min_n, random_state=seed) for _, g in groups], ignore_index=True)


def external_frame(external_features: Path, external_test: Path, cols: list[str]) -> pd.DataFrame:
    feat = pd.read_csv(external_features)
    meta = pd.read_csv(external_test)
    return meta.merge(feat[["id", *cols]], on="id", how="left", validate="one_to_one")


def build_train_defs(split_frames: dict[str, pd.DataFrame], seed: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    train = {s: split_frames[f"{s}_train"] for s in SOURCES}
    dev = {s: split_frames[f"{s}_dev"] for s in SOURCES}
    combined_train = pd.concat([train[s] for s in SOURCES], ignore_index=True)
    combined_dev = pd.concat([dev[s] for s in SOURCES], ignore_index=True)
    return {
        "train_ghostbuster": (train["ghostbuster"], dev["ghostbuster"]),
        "train_m4": (train["m4"], dev["m4"]),
        "train_hc3_plus": (train["hc3_plus"], dev["hc3_plus"]),
        "train_combined_strict": (combined_train, combined_dev),
        "train_balanced_combined_strict": (balanced_source_label_sample(combined_train, seed), combined_dev),
        "train_leave_out_ghostbuster": (pd.concat([train["m4"], train["hc3_plus"]], ignore_index=True), pd.concat([dev["m4"], dev["hc3_plus"]], ignore_index=True)),
        "train_leave_out_m4": (pd.concat([train["ghostbuster"], train["hc3_plus"]], ignore_index=True), pd.concat([dev["ghostbuster"], dev["hc3_plus"]], ignore_index=True)),
        "train_leave_out_hc3_plus": (pd.concat([train["ghostbuster"], train["m4"]], ignore_index=True), pd.concat([dev["ghostbuster"], dev["m4"]], ignore_index=True)),
    }


def bootstrap_ci_for_dir(out_dir: Path, n_boot: int, seed: int) -> dict:
    arr = np.load(out_dir / "bootstrap_arrays.npz")
    y = arr["y"]
    prob = arr["prob"]
    pred = arr["pred"]
    rng = np.random.default_rng(seed)
    vals = {"auroc": [], "auprc": [], "f1": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp, pr = y[idx], prob[idx], pred[idx]
        if len(np.unique(yy)) > 1:
            vals["auroc"].append(roc_auc_score(yy, pp))
            vals["auprc"].append(average_precision_score(yy, pp))
        vals["f1"].append(f1_score(yy, pr, zero_division=0))
    out = {}
    for metric, metric_vals in vals.items():
        if metric_vals:
            out[f"{metric}_ci_low"] = float(np.percentile(metric_vals, 2.5))
            out[f"{metric}_ci_high"] = float(np.percentile(metric_vals, 97.5))
        else:
            out[f"{metric}_ci_low"] = np.nan
            out[f"{metric}_ci_high"] = np.nan
    return out


def distribution_shift_report(train_df: pd.DataFrame, test_sets: dict[str, pd.DataFrame], cols: list[str], output_dir: Path) -> None:
    x_train, _, _ = make_xy(train_df, cols)
    rows = []
    summary = {"created_at": now(), "test_sets": {}}
    train_mean = x_train.mean()
    train_var = x_train.var(ddof=1)
    train_std = x_train.std(ddof=1).replace(0, np.nan)
    for test_name, df in test_sets.items():
        x_test, _, _ = make_xy(df, cols, x_train.median(numeric_only=True).fillna(0.0).to_dict())
        smd = ((x_test.mean() - train_mean) / train_std).abs().replace([np.inf, -np.inf], np.nan)
        ks_vals = []
        for col in cols:
            try:
                ks = float(ks_2samp(x_train[col], x_test[col], nan_policy="omit").statistic)
            except Exception:
                ks = np.nan
            ks_vals.append(ks)
            rows.append(
                {
                    "test_name": test_name,
                    "feature": col,
                    "train_mean": float(train_mean[col]),
                    "train_variance": float(train_var[col]),
                    "test_mean": float(x_test[col].mean()),
                    "test_variance": float(x_test[col].var(ddof=1)),
                    "standardized_mean_difference": float(smd[col]) if pd.notna(smd[col]) else np.nan,
                    "ks_statistic": ks,
                }
            )
        rep = pd.DataFrame([r for r in rows if r["test_name"] == test_name])
        top = rep.sort_values(["standardized_mean_difference", "ks_statistic"], ascending=False).head(30)
        summary["test_sets"][test_name] = {
            "mean_smd": float(rep["standardized_mean_difference"].mean()),
            "mean_ks": float(rep["ks_statistic"].mean()),
            "top_30_shifted_features": top[["feature", "standardized_mean_difference", "ks_statistic"]].to_dict(orient="records"),
        }
    write_csv(pd.DataFrame(rows), output_dir / "distribution_shift_report.csv")
    save_json(summary, output_dir / "distribution_shift_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--external_features", required=True)
    parser.add_argument("--source_splits", required=True)
    parser.add_argument("--external_test", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_n", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    feature_df = pd.read_csv(args.features)
    clean_cols, clean_info = select_clean_columns(feature_df)
    split_frames, missing_df = extract_feature_splits(Path(args.features), Path(args.source_splits), clean_cols)
    test_sets = {
        "ghostbuster_strict_test": split_frames["ghostbuster_test"],
        "m4_strict_test": split_frames["m4_test"],
        "hc3_plus_strict_test": split_frames["hc3_plus_test"],
        "all_samples": external_frame(Path(args.external_features), Path(args.external_test), clean_cols),
    }
    train_defs = build_train_defs(split_frames, args.seed)
    rows = []
    leakage_rows = []
    for train_name, (train_df, dev_df) in train_defs.items():
        train_ids = set(train_df["id"].astype(str))
        dev_ids = set(dev_df["id"].astype(str))
        leakage_rows.append({"train_name": train_name, "dev_overlap_ids": len(train_ids & dev_ids)})
        model, best_name, medians, meta = train_one(train_name, train_df, dev_df, clean_cols, checkpoint_dir)
        for test_name, test_df in test_sets.items():
            test_ids = set(test_df["id"].astype(str))
            leakage_rows.append({"train_name": train_name, "test_name": test_name, "train_test_overlap_ids": len(train_ids & test_ids)})
            row = save_eval_outputs(
                model,
                train_name,
                test_name,
                test_df,
                clean_cols,
                medians,
                output_dir / f"{train_name}_to_{test_name}",
                best_name,
                int(len(train_df)),
            )
            rows.append(row)
    matrix = pd.DataFrame(rows)
    write_csv(matrix, output_dir / "source_generalization_matrix.csv")
    for metric in ["auroc", "auprc", "tpr_at_fpr_1pct", "ece"]:
        pivot = matrix.pivot(index="train_name", columns="test_name", values=metric)
        write_csv(pivot.reset_index(), output_dir / f"source_generalization_matrix_{metric}.csv")
    ci_rows = []
    for row in rows:
        row_ci = dict(row)
        row_ci.update(bootstrap_ci_for_dir(output_dir / f"{row['train_name']}_to_{row['test_name']}", args.bootstrap_n, args.seed))
        ci_rows.append(row_ci)
    write_csv(pd.DataFrame(ci_rows), output_dir / "source_generalization_matrix_with_ci.csv")
    combined_train = train_defs["train_combined_strict"][0]
    distribution_shift_report(combined_train, {
        "ghostbuster_test": test_sets["ghostbuster_strict_test"],
        "m4_test": test_sets["m4_strict_test"],
        "hc3_plus_test": test_sets["hc3_plus_strict_test"],
        "all_samples": test_sets["all_samples"],
    }, clean_cols, output_dir)
    save_json(
        {
            "created_at": now(),
            "features": args.features,
            "external_features": args.external_features,
            "source_splits": args.source_splits,
            "external_test": args.external_test,
            "cleaning_info": clean_info,
            "missing_id_count": int(len(missing_df)),
            "leakage_checks": leakage_rows,
            "bootstrap_n": args.bootstrap_n,
        },
        output_dir / "source_matrix_manifest.json",
    )
    write_csv(pd.DataFrame(leakage_rows), output_dir / "leakage_check.csv")
    print(f"Wrote source generalization matrix to {output_dir / 'source_generalization_matrix.csv'}")


if __name__ == "__main__":
    main()
