#!/usr/bin/env python3
"""Smoke experiment for abstract transition-state profiling features."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.feature_transition_profile import build_from_cache, build_smoke_features, discover_token_loss_cache
from src.train_eval import detector_metrics, probabilities, save_calibration_curve, save_pr_curve, save_roc_curve
from src.utils import write_csv

sns.set_theme(style="whitegrid", context="talk")

META = {"id", "text", "label", "source_dataset", "language", "domain", "generator", "attack_type", "split", "type", "source", "topic"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_fig(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def cleaned_full_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if "32b" not in c.lower()]
    allowed = [c for c in cols if c.startswith(("burst_", "struct_", "scale_", "qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_"))]
    work = df[allowed].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    constant = work.nunique(dropna=True) <= 1
    all_nan = work.isna().all()
    return [c for c in allowed if not constant.get(c, False) and not all_nan.get(c, False)]


def make_xy(df: pd.DataFrame, cols: list[str], medians: dict | None = None):
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(medians).fillna(0.0)
    y = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return x, y, {k: float(v) for k, v in medians.items()}


def candidate_models():
    return {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, class_weight="balanced", n_jobs=-1),
    }


def fit_select(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str]):
    x_train, y_train, med = make_xy(train_df, cols)
    x_dev, y_dev, _ = make_xy(dev_df, cols, med)
    best = None
    rows = []
    for name, model in candidate_models().items():
        model.fit(x_train, y_train)
        prob = probabilities(model, x_dev)
        pred = (prob >= 0.5).astype(int)
        metrics = {"model": name}
        metrics.update(detector_metrics(y_dev, prob, y_pred=pred))
        rows.append(metrics)
        rank = (metrics.get("auprc", -np.inf), metrics.get("auroc", -np.inf), metrics.get("f1", -np.inf))
        if best is None or rank > best[0]:
            best = (rank, name, model, med, rows)
    return best[1], best[2], best[3], pd.DataFrame(rows)


def eval_model(model, test_df: pd.DataFrame, cols: list[str], med: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, _ = make_xy(test_df, cols, med)
    prob = probabilities(model, x)
    pred = (prob >= 0.5).astype(int)
    metrics = detector_metrics(y, prob, y_pred=pred)
    write_csv(pd.DataFrame([metrics]), out_dir / "metrics.csv")
    keep = [c for c in ["id", "text", "label", "source_dataset", "domain", "generator", "split"] if c in test_df.columns]
    preds = test_df[keep].copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    write_csv(preds, out_dir / "predictions.csv")
    write_csv(save_roc_curve(y, prob, out_dir), out_dir / "roc_curve.csv")
    write_csv(save_pr_curve(y, prob, out_dir), out_dir / "pr_curve.csv")
    write_csv(save_calibration_curve(y, prob, out_dir), out_dir / "calibration_bins.csv")
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(classification_report(y, pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
    return metrics


def plot_embedding(df: pd.DataFrame, cols: list[str], plot_dir: Path) -> dict[str, str]:
    x, y, _ = make_xy(df, cols)
    if len(df) < 4 or len(cols) < 2:
        return {"pca_note": "not enough rows/features"}
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    plot_df = df.copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    plot_df["label_name"] = y.map({0: "Human", 1: "AI"})
    source_col = "source_dataset" if "source_dataset" in plot_df.columns else "split"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="label_name", style=source_col, s=55, alpha=0.8, ax=ax)
    ax.set_title("Transition-profile PCA by label")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    save_fig(fig, plot_dir / "transition_pca_by_label")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=source_col, style="label_name", s=55, alpha=0.8, ax=ax)
    ax.set_title("Transition-profile PCA by source")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    save_fig(fig, plot_dir / "transition_pca_by_source")
    return {
        "pca_explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
    }


def source_separability_proxy(df: pd.DataFrame, cols: list[str]) -> dict:
    if len(df) < 10:
        return {"note": "too few rows"}
    x, y, med = make_xy(df, cols)
    source = df.get("source_dataset", pd.Series(["unknown"] * len(df))).astype(str)
    out = {}
    try:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=42, stratify=y if y.nunique() > 1 else None)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xtr, ytr)
        out["label_probe_accuracy"] = float(clf.score(xte, yte))
    except Exception as exc:
        out["label_probe_error"] = str(exc)
    try:
        if source.nunique() > 1:
            xtr, xte, strn, ste = train_test_split(x, source, test_size=0.35, random_state=42, stratify=source if source.value_counts().min() >= 2 else None)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xtr, strn)
            out["source_probe_accuracy"] = float(clf.score(xte, ste))
    except Exception as exc:
        out["source_probe_error"] = str(exc)
    return out


def build_transition_inputs(train_csv: Path, external_csv: Path, sample_train: int, sample_external: int, seed: int) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    ext = pd.read_csv(external_csv)
    train = train.groupby("label", group_keys=False).apply(lambda g: g.sample(n=min(len(g), max(1, sample_train // max(train["label"].nunique(), 1))), random_state=seed))
    ext = ext.groupby("label", group_keys=False).apply(lambda g: g.sample(n=min(len(g), max(1, sample_external // max(ext["label"].nunique(), 1))), random_state=seed))
    train["transition_split"] = "combined_public_smoke"
    ext["transition_split"] = "all_samples_smoke"
    return pd.concat([train, ext], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train_sets/combined_public_train.csv")
    parser.add_argument("--external_csv", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--output_feature_dir", default="features_transition")
    parser.add_argument("--output_result_dir", default="results_transition")
    parser.add_argument("--sample_train", type=int, default=60)
    parser.add_argument("--sample_external", type=int, default=40)
    parser.add_argument("--model_key", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_dir = ROOT / args.output_feature_dir
    result_dir = ROOT / args.output_result_dir
    plot_dir = result_dir / "plots"
    feature_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    smoke_input = feature_dir / "transition_smoke_input.csv"
    transition_csv = feature_dir / "transition_features.csv"

    cache_paths = [
        ROOT / "features_by_dataset",
        ROOT / "features_external",
        ROOT / "features",
        ROOT / "features_transition",
    ]
    cache, seq_cols = discover_token_loss_cache(cache_paths)
    token_cache_found = cache is not None
    if token_cache_found:
        transition_df = build_from_cache(cache, seq_cols, transition_csv)
        meta = pd.concat([pd.read_csv(ROOT / args.train_csv), pd.read_csv(ROOT / args.external_csv)], ignore_index=True)
    else:
        meta = build_transition_inputs(ROOT / args.train_csv, ROOT / args.external_csv, args.sample_train, args.sample_external, args.seed)
        meta.to_csv(smoke_input, index=False)
        transition_df = build_smoke_features(smoke_input, transition_csv, model_key=args.model_key, sample_size=0, seed=args.seed, max_length=args.max_length)

    merged = meta.merge(transition_df, on="id", how="inner")
    trans_cols = [c for c in transition_df.columns if c != "id" and pd.api.types.is_numeric_dtype(transition_df[c])]
    plot_info = plot_embedding(merged, trans_cols, plot_dir)
    probe_info = source_separability_proxy(merged, trans_cols)

    # Ablation uses only rows where transition features exist. This is a smoke
    # test when token-level caches are unavailable.
    full_train = pd.read_csv(ROOT / args.full_features)
    full_external = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(full_train)
    combined_full = pd.concat([
        pd.read_csv(ROOT / args.train_csv).merge(full_train[["id", *full_cols]], on="id", how="left"),
        pd.read_csv(ROOT / args.external_csv).merge(full_external[["id", *full_cols]], on="id", how="left"),
    ], ignore_index=True)
    ablation_base = merged[["id", "label", "transition_split"] + trans_cols].merge(combined_full[["id", *full_cols]], on="id", how="left")
    train_df = ablation_base[ablation_base["transition_split"].eq("combined_public_smoke")]
    external_df = ablation_base[ablation_base["transition_split"].eq("all_samples_smoke")]
    if len(train_df) >= 10:
        dev_df = train_df.groupby("label", group_keys=False).apply(lambda g: g.sample(n=max(1, int(len(g) * 0.25)), random_state=args.seed))
        train_fit = train_df.drop(dev_df.index)
    else:
        train_fit = train_df
        dev_df = train_df
    experiments = {
        "transition_only": trans_cols,
        "full_cleaned_without_transition": full_cols,
        "full_cleaned_plus_transition": full_cols + trans_cols,
    }
    rows = []
    for name, cols in experiments.items():
        if len(train_fit) < 4 or len(set(train_fit["label"])) < 2 or len(set(dev_df["label"])) < 2:
            rows.append({"experiment": name, "status": "skipped_too_few_rows", "n_features": len(cols)})
            continue
        best_name, model, med, dev_metrics = fit_select(train_fit, dev_df, cols)
        same_metrics = eval_model(model, dev_df, cols, med, result_dir / f"{name}_same_source_smoke")
        ext_metrics = eval_model(model, external_df, cols, med, result_dir / f"{name}_all_samples_smoke") if len(external_df) else {}
        joblib.dump(model, result_dir / f"{name}_best_model.joblib")
        for split_name, metrics in [("same_source_smoke", same_metrics), ("all_samples_smoke", ext_metrics)]:
            rows.append(
                {
                    "experiment": name,
                    "test_set": split_name,
                    "status": "ok",
                    "best_model": best_name,
                    "n_train": int(len(train_fit)),
                    "n_test": int(len(dev_df) if split_name == "same_source_smoke" else len(external_df)),
                    "n_features": int(len(cols)),
                    "auroc": metrics.get("auroc", np.nan),
                    "auprc": metrics.get("auprc", np.nan),
                    "f1": metrics.get("f1", np.nan),
                    "tpr_at_fpr_1pct": metrics.get("tpr_at_fpr_1pct", np.nan),
                    "tpr_at_fpr_5pct": metrics.get("tpr_at_fpr_5pct", np.nan),
                    "ece": metrics.get("expected_calibration_error", metrics.get("ECE", np.nan)),
                }
            )
    summary = pd.DataFrame(rows)
    write_csv(summary, result_dir / "transition_ablation_summary.csv")
    save_json(
        {
            "created_at": now(),
            "token_level_loss_cache_found": token_cache_found,
            "token_level_loss_cache": str(cache) if cache else None,
            "sequence_columns": seq_cols,
            "transition_feature_file": str(transition_csv),
            "n_transition_rows": int(len(transition_df)),
            "n_transition_features": int(len(trans_cols)),
            "model_key_used_for_smoke": None if token_cache_found else args.model_key,
            "smoke_test": bool(not token_cache_found),
            "plot_info": plot_info,
            "probe_info": probe_info,
            "notes": "No raw token-id transitions are used; token losses are mapped to abstract loss-quantile and probability-rank states.",
        },
        result_dir / "transition_experiment_manifest.json",
    )
    print(f"Wrote {transition_csv}")
    print(f"Wrote {result_dir / 'transition_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
