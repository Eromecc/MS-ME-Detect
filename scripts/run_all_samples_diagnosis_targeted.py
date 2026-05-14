#!/usr/bin/env python3
"""All-samples shift diagnosis, M4-targeted variants, and presentation figures."""

from __future__ import annotations

import argparse
import json
import re
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
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_source_matrix_eval import (  # noqa: E402
    SOURCES,
    SPLITS,
    build_train_defs,
    candidate_models,
    external_frame,
    make_xy,
    model_ranking_tuple,
    probabilities,
    save_eval_outputs,
    select_clean_columns,
)
from src import config  # noqa: E402
from src.train_eval import detector_metrics  # noqa: E402
from src.utils import save_json, write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def save_fig(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_csv_warn(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"missing input, skipping: {path}")
        return None
    return pd.read_csv(path)


def load_split_features(source_splits: Path, features: Path, clean_cols: list[str]) -> dict[str, pd.DataFrame]:
    feature_df = pd.read_csv(features)
    feature_by_id = feature_df[["id", *clean_cols]].copy()
    frames = {}
    for source in SOURCES:
        for split in SPLITS:
            meta = pd.read_csv(source_splits / f"{source}_strict_{split}.csv")
            meta["label"] = pd.to_numeric(meta["label"], errors="coerce").astype(int)
            frames[f"{source}_{split}"] = meta.merge(feature_by_id, on="id", how="left", validate="one_to_one")
    return frames


def prediction_dirs() -> dict[str, Path]:
    base = ROOT / "results_source_matrix"
    mapping = {
        "ghostbuster": base / "train_ghostbuster_to_all_samples",
        "m4": base / "train_m4_to_all_samples",
        "hc3_plus": base / "train_hc3_plus_to_all_samples",
        "combined_strict": base / "train_combined_strict_to_all_samples",
        "balanced_combined": base / "train_balanced_combined_strict_to_all_samples",
        "leave_out_ghostbuster": base / "train_leave_out_ghostbuster_to_all_samples",
    }
    return mapping


def error_analysis(external_features: Path, external_test: Path, diagnosis_dir: Path, plots_dir: Path) -> pd.DataFrame:
    test = pd.read_csv(external_test)
    feats = pd.read_csv(external_features)
    prob_cols = [c for c in feats.columns if c.startswith(("qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_"))]
    feat_subset = feats[["id", *prob_cols]].copy()
    rows = []
    examples_md = []
    all_examples = []
    for model, out_dir in prediction_dirs().items():
        pred_path = out_dir / "predictions.csv"
        met_path = out_dir / "metrics.csv"
        if not pred_path.exists():
            warnings.warn(f"missing predictions for {model}: {pred_path}")
            continue
        pred = pd.read_csv(pred_path).merge(test[["id", "text", "label", "domain", "generator"]], on="id", how="left", suffixes=("", "_test"))
        pred = pred.merge(feat_subset, on="id", how="left")
        if "label_test" in pred.columns:
            pred["label"] = pred["label"].fillna(pred["label_test"])
        pred["label"] = pd.to_numeric(pred["label"], errors="coerce").astype(int)
        pred["text_length"] = pred["text"].astype(str).str.len()
        pred["correct"] = pred["prediction"].astype(int).eq(pred["label"])
        pred["error_type"] = np.select(
            [
                (pred["label"].eq(1) & pred["prediction"].eq(1)),
                (pred["label"].eq(0) & pred["prediction"].eq(1)),
                (pred["label"].eq(0) & pred["prediction"].eq(0)),
                (pred["label"].eq(1) & pred["prediction"].eq(0)),
            ],
            ["TP", "FP", "TN", "FN"],
            default="NA",
        )
        wrong = pred[~pred["correct"]].copy()
        wrong["wrong_confidence"] = np.where(wrong["prediction"].eq(1), wrong["ai_probability"], 1 - wrong["ai_probability"])
        top_wrong = wrong.sort_values("wrong_confidence", ascending=False).head(30)
        top_wrong.to_csv(diagnosis_dir / f"error_examples_{model}.csv", index=False)
        with open(diagnosis_dir / f"error_examples_{model}.md", "w", encoding="utf-8") as handle:
            handle.write(f"# Confident wrong predictions: {model}\n\n")
            for _, r in top_wrong.iterrows():
                handle.write(f"- id={r['id']} true={r['label']} pred={r['prediction']} prob={r['ai_probability']:.4f} domain={r.get('domain','')} generator={r.get('generator','')}\n")
        top_wrong["model"] = model
        all_examples.append(top_wrong)
        counts = pred["error_type"].value_counts().to_dict()
        human_prob = pred.loc[pred["label"].eq(0), "ai_probability"]
        ai_prob = pred.loc[pred["label"].eq(1), "ai_probability"]
        metrics = pd.read_csv(met_path).iloc[0].to_dict() if met_path.exists() else {}
        rows.append(
            {
                "model": model,
                "n": len(pred),
                "TP": counts.get("TP", 0),
                "FP": counts.get("FP", 0),
                "TN": counts.get("TN", 0),
                "FN": counts.get("FN", 0),
                "human_prob_mean": human_prob.mean(),
                "human_prob_median": human_prob.median(),
                "human_prob_q10": human_prob.quantile(0.10),
                "human_prob_q90": human_prob.quantile(0.90),
                "ai_prob_mean": ai_prob.mean(),
                "ai_prob_median": ai_prob.median(),
                "ai_prob_q10": ai_prob.quantile(0.10),
                "ai_prob_q90": ai_prob.quantile(0.90),
                "prob_direction_reversed": bool(ai_prob.mean() < human_prob.mean()),
                "wrong_text_length_mean": wrong["text_length"].mean(),
                "wrong_probability_feature_mean": wrong[prob_cols].mean(numeric_only=True).mean() if prob_cols else np.nan,
                "auroc": metrics.get("auroc", np.nan),
                "auprc": metrics.get("auprc", np.nan),
                "f1": metrics.get("f1", np.nan),
            }
        )
        plot_probability_diagnostics(model, pred, metrics, plots_dir)
    out = pd.DataFrame(rows)
    write_csv(out, diagnosis_dir / "all_samples_error_analysis.csv")
    save_json({"created_at": now(), "models": rows}, diagnosis_dir / "all_samples_error_summary.json")
    if all_examples:
        write_csv(pd.concat(all_examples, ignore_index=True), diagnosis_dir / "all_samples_confident_wrong_predictions.csv")
    return out


def plot_probability_diagnostics(model: str, pred: pd.DataFrame, metrics: dict, plots_dir: Path) -> None:
    label_name = pred["label"].map({0: "Human", 1: "AI"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(data=pred, x="ai_probability", hue=label_name, bins=24, stat="density", common_norm=False, ax=axes[0])
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title("Histogram")
    sns.kdeplot(data=pred, x="ai_probability", hue=label_name, common_norm=False, ax=axes[1], warn_singular=False)
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Density")
    sns.violinplot(data=pred.assign(label_name=label_name), x="label_name", y="ai_probability", ax=axes[2], cut=0)
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1.5)
    axes[2].set_title("Violin")
    fig.suptitle(f"{model} on all_samples | AUROC={metrics.get('auroc', np.nan):.3f} AUPRC={metrics.get('auprc', np.nan):.3f} F1={metrics.get('f1', np.nan):.3f}")
    save_fig(fig, plots_dir / f"all_samples_probability_distribution_{safe_name(model)}")

    cm = confusion_matrix(pred["label"], pred["prediction"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model}")
    save_fig(fig, plots_dir / f"confusion_matrix_{safe_name(model)}")

    tmp = pred.copy()
    tmp["length_bin"] = pd.qcut(tmp["text_length"], q=5, duplicates="drop")
    err = tmp.groupby("length_bin", observed=False)["correct"].apply(lambda s: 1 - s.mean()).reset_index(name="error_rate")
    err["length_bin"] = err["length_bin"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=err, x="length_bin", y="error_rate", color="#4C78A8", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Text length bin")
    ax.set_ylabel("Error rate")
    ax.set_title(f"Error Rate by Text Length: {model}")
    save_fig(fig, plots_dir / f"error_rate_by_length_{safe_name(model)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=pred, x="text_length", y="ai_probability", hue=label_name, style=pred["correct"], alpha=0.75, ax=ax)
    ax.axhline(0.5, color="black", linestyle="--")
    ax.set_title(f"Probability vs Text Length: {model}")
    ax.set_xlabel("Text length")
    ax.set_ylabel("AI probability")
    save_fig(fig, plots_dir / f"probability_vs_length_{safe_name(model)}")


def train_targeted_variants(features: Path, source_splits: Path, external_features: Path, external_test: Path, checkpoint_dir: Path, results_dir: Path, seed: int) -> pd.DataFrame:
    feature_df = pd.read_csv(features)
    clean_cols, clean_info = select_clean_columns(feature_df)
    frames = load_split_features(source_splits, features, clean_cols)
    ext = external_frame(external_features, external_test, clean_cols)
    m4_train = frames["m4_train"]
    m4_dev = frames["m4_dev"]
    hc3_train = frames["hc3_plus_train"]
    hc3_dev = frames["hc3_plus_dev"]
    variants = {
        "m4_only_cleaned": (m4_train, m4_dev),
        "m4_domain_balanced": (balanced_sample(m4_train, ["domain"], seed), m4_dev),
        "m4_generator_balanced": (balanced_sample(m4_train, ["generator"], seed), m4_dev),
        "m4_label_balanced": (balanced_sample(m4_train, ["label"], seed), m4_dev),
        "m4_plus_hc3": (pd.concat([m4_train, hc3_train], ignore_index=True), pd.concat([m4_dev, hc3_dev], ignore_index=True)),
        "m4_plus_hc3_balanced": (balanced_sample(pd.concat([m4_train, hc3_train], ignore_index=True), ["source_dataset", "label"], seed), pd.concat([m4_dev, hc3_dev], ignore_index=True)),
    }
    variants["m4_plus_hc3_without_extreme_domains"] = (drop_extreme_domains(variants["m4_plus_hc3"][0], clean_cols), variants["m4_plus_hc3"][1])
    rows = []
    for name, (train_df, dev_df) in variants.items():
        model, best_name, medians, meta = train_select(name, train_df, dev_df, clean_cols, checkpoint_dir)
        out_dir = results_dir / name
        row = save_eval_outputs(model, name, "all_samples", ext, clean_cols, medians, out_dir, best_name, len(train_df))
        row.update(
            {
                "train_variant": name,
                "n_human": int((train_df["label"] == 0).sum()),
                "n_ai": int((train_df["label"] == 1).sum()),
                "domain_distribution": json.dumps(train_df["domain"].value_counts().to_dict(), sort_keys=True),
                "generator_distribution": json.dumps(train_df["generator"].value_counts().to_dict(), sort_keys=True),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    cols = ["train_variant", "n_train", "n_human", "n_ai", "domain_distribution", "generator_distribution", "best_model", "n_features", "auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score"]
    write_csv(summary[cols], results_dir / "m4_targeted_summary.csv")
    save_json({"created_at": now(), "cleaning_info": clean_info}, results_dir / "m4_targeted_manifest.json")
    return summary[cols]


def balanced_sample(df: pd.DataFrame, group_cols: list[str], seed: int) -> pd.DataFrame:
    groups = [(k, g) for k, g in df.groupby(group_cols, dropna=False) if len(g)]
    if not groups:
        return df.copy()
    min_n = min(len(g) for _, g in groups)
    return pd.concat([g.sample(n=min_n, random_state=seed) for _, g in groups], ignore_index=True)


def drop_extreme_domains(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if "domain" not in df.columns or df["domain"].nunique(dropna=True) <= 2:
        return df.copy()
    x, _, _ = make_xy(df, cols)
    global_mean = x.mean()
    global_std = x.std(ddof=1).replace(0, np.nan)
    scores = {}
    for domain, idx in df.groupby("domain").groups.items():
        if len(idx) < 30:
            scores[domain] = np.inf
        else:
            scores[domain] = float(((x.loc[idx].mean() - global_mean) / global_std).abs().mean())
    threshold = np.nanpercentile([v for v in scores.values() if np.isfinite(v)], 90)
    keep_domains = [d for d, s in scores.items() if np.isfinite(s) and s <= threshold]
    out = df[df["domain"].isin(keep_domains)].copy()
    return out if len(out) >= 100 else df.copy()


def train_select(name: str, train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str], checkpoint_dir: Path):
    x_train, y_train, medians = make_xy(train_df, cols)
    x_dev, y_dev, _ = make_xy(dev_df, cols, medians)
    rows = []
    best = None
    for model_name, model in candidate_models(y_train):
        fitted = model
        fitted.fit(x_train, y_train)
        prob = probabilities(fitted, x_dev)
        pred = (prob >= 0.5).astype(int)
        metrics = {"model": model_name}
        metrics.update(detector_metrics(y_dev, prob, y_pred=pred, threshold=0.5))
        rows.append(metrics)
        if best is None or model_ranking_tuple(metrics) > model_ranking_tuple(best[0]):
            best = (metrics, fitted, model_name)
    best_metrics, best_model, best_name = best
    ckpt = checkpoint_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ckpt / "best_model.joblib")
    save_json(cols, ckpt / "feature_columns.json")
    save_json(medians, ckpt / "feature_medians.json")
    save_json({"created_at": now(), "train_variant": name, "best_model_name": best_name, "n_train": len(train_df), "n_dev": len(dev_df), "n_features": len(cols), "dev_metrics": best_metrics, "candidate_dev_metrics": rows}, ckpt / "train_metadata.json")
    return best_model, best_name, medians, best_metrics


def plot_heatmaps(source_matrix: Path, plots_dir: Path) -> None:
    df = read_csv_warn(source_matrix)
    if df is None:
        return
    metrics = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1"), ("tpr_at_fpr_1pct", "TPR@FPR=1%"), ("ece", "ECE")]
    for metric, title in metrics:
        pivot = df.pivot(index="train_name", columns="test_name", values=metric)
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r" if metric == "ece" else "viridis", ax=ax, cbar_kws={"label": title})
        ax.set_title(f"Source Generalization Matrix: {title}")
        ax.set_xlabel("Test set")
        ax.set_ylabel("Train set")
        if "all_samples" in pivot.columns:
            j = list(pivot.columns).index("all_samples")
            ax.add_patch(plt.Rectangle((j, 0), 1, len(pivot.index), fill=False, edgecolor="red", linewidth=3))
        for source in ["ghostbuster", "m4", "hc3_plus"]:
            train = f"train_{source}"
            test = f"{source}_strict_test"
            if train in pivot.index and test in pivot.columns:
                i = list(pivot.index).index(train)
                j = list(pivot.columns).index(test)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="white", linewidth=3))
        save_fig(fig, plots_dir / f"source_matrix_{metric}_heatmap")


def plot_public_drop(source_matrix: Path, plots_dir: Path) -> None:
    df = read_csv_warn(source_matrix)
    if df is None:
        return
    for metric in ["auroc", "auprc", "f1"]:
        public = df[df["test_name"].ne("all_samples")].groupby("train_name")[metric].mean()
        external = df[df["test_name"].eq("all_samples")].set_index("train_name")[metric]
        common = public.index.intersection(external.index)
        plot_df = pd.DataFrame({"train_name": common, "public_mean": public.loc[common].values, "all_samples": external.loc[common].values})
        plot_df["drop"] = plot_df["public_mean"] - plot_df["all_samples"]
        plot_df = plot_df.sort_values("drop", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        y = np.arange(len(plot_df))
        ax.hlines(y, plot_df["all_samples"], plot_df["public_mean"], color="#999999", linewidth=2)
        ax.scatter(plot_df["public_mean"], y, label="Public tests mean", s=80)
        ax.scatter(plot_df["all_samples"], y, label="all_samples", s=80)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["train_name"])
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Public Test to all_samples Performance Drop: {metric.upper()}")
        ax.legend()
        save_fig(fig, plots_dir / f"public_to_all_samples_drop_{metric}")


def plot_shift_figures(report_path: Path, summary_path: Path, source_plots: Path) -> list[str]:
    generated = []
    rep = read_csv_warn(report_path)
    if rep is None or not summary_path.exists():
        return generated
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["test_sets"]
    smd_df = pd.DataFrame([{"test_name": k, "mean_smd": v["mean_smd"], "mean_ks": v["mean_ks"]} for k, v in summary.items()])
    for metric, ylabel, out in [("mean_smd", "Mean standardized mean difference", "mean_smd_by_test_set"), ("mean_ks", "Mean KS statistic", "mean_ks_by_test_set")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=smd_df, x="test_name", y=metric, color="#4C78A8", ax=ax)
        ax.set_xlabel("Test set")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by Test Set")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        save_fig(fig, source_plots / out)
        generated.append(str((source_plots / out).with_suffix(".png")))
    top = rep[rep["test_name"].eq("all_samples")].sort_values("standardized_mean_difference", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=top, y="feature", x="standardized_mean_difference", color="#F58518", ax=ax)
    ax.set_xlabel("Standardized mean difference")
    ax.set_ylabel("Feature")
    ax.set_title("Top Shifted Features: all_samples vs Combined Public Train")
    save_fig(fig, source_plots / "all_samples_top_shifted_features")
    generated.append(str((source_plots / "all_samples_top_shifted_features").with_suffix(".png")))
    return generated


def plot_feature_distributions(features: Path, external_features: Path, external_test: Path, source_splits: Path, diagnosis_plots: Path) -> None:
    feature_df = pd.read_csv(features)
    clean_cols, _ = select_clean_columns(feature_df)
    frames = load_split_features(source_splits, features, clean_cols)
    all_samples = external_frame(external_features, external_test, clean_cols)
    datasets = {
        "ghostbuster_test": frames["ghostbuster_test"],
        "m4_test": frames["m4_test"],
        "hc3_plus_test": frames["hc3_plus_test"],
        "all_samples": all_samples,
    }
    shift = read_csv_warn(ROOT / "results_source_matrix" / "distribution_shift_report.csv")
    top_features = ["scale_ppl_response_variance", "scale_bottom_10_percent_loss_mean_response_variance", "qwen25_1_5b_loss_min"]
    if shift is not None:
        top_features += shift[shift["test_name"].eq("all_samples")].sort_values("standardized_mean_difference", ascending=False)["feature"].head(30).tolist()
    seen = []
    top_features = [f for f in top_features if f in clean_cols and not (f in seen or seen.append(f))][:10]
    long_rows = []
    for name, df in datasets.items():
        for f in top_features:
            tmp = df[["label", f]].copy()
            tmp["dataset"] = name
            tmp["feature"] = f
            tmp["value"] = pd.to_numeric(tmp[f], errors="coerce")
            long_rows.append(tmp[["dataset", "label", "feature", "value"]])
    long = pd.concat(long_rows, ignore_index=True)
    for f in top_features:
        data = long[long["feature"].eq(f)].dropna()
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        sns.kdeplot(data=data, x="value", hue="dataset", common_norm=False, ax=axes[0], warn_singular=False)
        axes[0].set_title("Source-wise Density")
        sns.boxplot(data=data, x="dataset", y="value", ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right")
        axes[1].set_title("Source-wise Boxplot")
        sns.boxplot(data=data.assign(label_name=data["label"].map({0: "Human", 1: "AI"})), x="dataset", y="value", hue="label_name", ax=axes[2])
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=25, ha="right")
        axes[2].set_title("Human vs AI by Source")
        fig.suptitle(f"Feature Distribution: {f}")
        save_fig(fig, diagnosis_plots / f"feature_distribution_{safe_name(f)}")


def plot_roc_pr_comparison(diagnosis_plots: Path, targeted_summary: pd.DataFrame | None = None) -> None:
    models = prediction_dirs()
    if targeted_summary is not None and not targeted_summary.empty:
        best = targeted_summary.sort_values(["auprc", "auroc", "f1"], ascending=False).iloc[0]["train_variant"]
        models[f"targeted_best_{best}"] = ROOT / "results_targeted" / best
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    baseline = None
    for model, out_dir in models.items():
        pred_path = out_dir / "predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        y = pd.to_numeric(pred["label"], errors="coerce").astype(int)
        p = pd.to_numeric(pred["ai_probability"], errors="coerce")
        if y.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        auroc = auc(fpr, tpr)
        auprc = auc(rec, prec)
        baseline = y.mean()
        ax_roc.plot(fpr, tpr, label=f"{model} ({auroc:.3f})")
        ax_pr.plot(rec, prec, label=f"{model} ({auprc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC Curves on all_samples")
    ax_roc.legend(fontsize=9)
    if baseline is not None:
        ax_pr.axhline(baseline, color="black", linestyle="--", label=f"Random baseline ({baseline:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR Curves on all_samples")
    ax_pr.legend(fontsize=9)
    save_fig(fig_roc, diagnosis_plots / "all_samples_roc_curves_comparison")
    save_fig(fig_pr, diagnosis_plots / "all_samples_pr_curves_comparison")


def grouped_bar(df: pd.DataFrame, id_col: str, metrics: list[str], title: str, out: Path) -> None:
    keep = [id_col, *[m for m in metrics if m in df.columns]]
    if len(keep) <= 1:
        return
    long = df[keep].melt(id_vars=id_col, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=long, x=id_col, y="value", hue="metric", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(title="")
    save_fig(fig, out)


def plot_ablation_and_optimized() -> None:
    ab = read_csv_warn(ROOT / "results_ablation" / "combined_public_feature_ablation_summary.csv")
    if ab is not None:
        grouped_bar(ab, "feature_set", ["auroc", "auprc"], "Feature Ablation: AUROC and AUPRC", ROOT / "results_ablation" / "combined_public_ablation_auroc_auprc")
        grouped_bar(ab, "feature_set", ["tpr_at_fpr_1pct", "tpr_at_fpr_5pct"], "Feature Ablation: Low-FPR TPR", ROOT / "results_ablation" / "combined_public_ablation_low_fpr")
        grouped_bar(ab, "feature_set", ["ece", "brier_score"], "Feature Ablation: Calibration", ROOT / "results_ablation" / "combined_public_ablation_calibration")
    comp = read_csv_warn(ROOT / "results_optimized" / "combined_public_cleaned_tuned_comparison.csv")
    if comp is not None:
        plots = ROOT / "results_optimized" / "plots"
        grouped_bar(comp, "model_version", ["auroc", "auprc", "f1"], "Original vs Cleaned vs Tuned Performance", plots / "cleaned_tuned_performance_comparison")
        grouped_bar(comp, "model_version", ["ece", "brier_score"], "Original vs Cleaned vs Tuned Calibration", plots / "cleaned_tuned_calibration_comparison")
        grouped_bar(comp, "model_version", ["tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct"], "Original vs Cleaned vs Tuned Low-FPR Metrics", plots / "cleaned_tuned_low_fpr_comparison")


def plot_targeted(summary: pd.DataFrame, plots_dir: Path) -> None:
    grouped_bar(summary, "train_variant", ["auroc", "auprc", "f1"], "M4-targeted Variants on all_samples", plots_dir / "m4_targeted_performance")
    grouped_bar(summary, "train_variant", ["tpr_at_fpr_1pct", "tpr_at_fpr_5pct"], "M4-targeted Low-FPR TPR", plots_dir / "m4_targeted_low_fpr")
    grouped_bar(summary, "train_variant", ["ece", "brier_score"], "M4-targeted Calibration", plots_dir / "m4_targeted_calibration")
    rows = []
    for _, r in summary.iterrows():
        for kind, col in [("domain", "domain_distribution"), ("generator", "generator_distribution")]:
            try:
                dist = json.loads(r[col])
            except Exception:
                dist = {}
            for k, v in dist.items():
                rows.append({"train_variant": r["train_variant"], "kind": kind, "category": k, "n": v})
    dist_df = pd.DataFrame(rows)
    if not dist_df.empty:
        for kind in ["domain", "generator"]:
            piv = dist_df[dist_df["kind"].eq(kind)].pivot_table(index="train_variant", columns="category", values="n", fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 6))
            piv.plot(kind="bar", stacked=True, ax=ax, width=0.85)
            ax.set_title(f"M4-targeted Training {kind.title()} Distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Rows")
            ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
            save_fig(fig, plots_dir / f"m4_targeted_train_{kind}_distribution")
        # Required combined filename: use domain plot as concise overview.
        fig, ax = plt.subplots(figsize=(12, 6))
        dist_df.groupby(["train_variant", "kind"])["n"].sum().unstack(fill_value=0).plot(kind="bar", ax=ax)
        ax.set_title("M4-targeted Training Distribution Summary")
        ax.set_xlabel("")
        ax.set_ylabel("Rows")
        save_fig(fig, plots_dir / "m4_targeted_train_distribution")


def plot_feature_space(features: Path, external_features: Path, external_test: Path, source_splits: Path, diagnosis_plots: Path, seed: int) -> None:
    feature_df = pd.read_csv(features)
    clean_cols, _ = select_clean_columns(feature_df)
    frames = load_split_features(source_splits, features, clean_cols)
    all_samples = external_frame(external_features, external_test, clean_cols)
    parts = []
    for name in ["ghostbuster_test", "m4_test", "hc3_plus_test"]:
        tmp = frames[name].copy()
        tmp["plot_source"] = name
        parts.append(tmp)
    tmp = all_samples.copy()
    tmp["plot_source"] = "all_samples"
    parts.append(tmp)
    df = pd.concat(parts, ignore_index=True)
    sample = df.groupby("plot_source", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 600), random_state=seed)).reset_index(drop=True)
    x, y, _ = make_xy(sample, clean_cols)
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
    sample["PC1"] = pca[:, 0]
    sample["PC2"] = pca[:, 1]
    for hue, out, title in [("plot_source", "feature_space_pca_by_source", "PCA Feature Space by Source"), ("label", "feature_space_pca_by_label", "PCA Feature Space by Label")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=sample, x="PC1", y="PC2", hue=hue, style=sample["plot_source"].eq("all_samples"), alpha=0.75, s=35, ax=ax)
        ax.set_title(title)
        save_fig(fig, diagnosis_plots / out)
    try:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=seed).fit_transform(x_scaled)
        sample["TSNE1"] = tsne[:, 0]
        sample["TSNE2"] = tsne[:, 1]
        for hue, out, title in [("plot_source", "feature_space_tsne_by_source", "t-SNE Feature Space by Source"), ("label", "feature_space_tsne_by_label", "t-SNE Feature Space by Label")]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=sample, x="TSNE1", y="TSNE2", hue=hue, style=sample["plot_source"].eq("all_samples"), alpha=0.75, s=35, ax=ax)
            ax.set_title(title)
            save_fig(fig, diagnosis_plots / out)
    except Exception as exc:
        warnings.warn(f"t-SNE failed, PCA figures retained: {exc}")


def model_comparison(targeted: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    comp = read_csv_warn(ROOT / "results_optimized" / "combined_public_cleaned_tuned_comparison.csv")
    if comp is not None:
        for _, r in comp.iterrows():
            rows.append({"model": r["model_version"], **{k: r.get(k, np.nan) for k in ["auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score"]}})
    sm = read_csv_warn(ROOT / "results_source_matrix" / "source_generalization_matrix.csv")
    if sm is not None:
        for train in ["train_m4", "train_leave_out_ghostbuster", "train_combined_strict", "train_balanced_combined_strict"]:
            hit = sm[(sm["train_name"].eq(train)) & (sm["test_name"].eq("all_samples"))]
            if not hit.empty:
                r = hit.iloc[0]
                rows.append({"model": train, **{k: r.get(k, np.nan) for k in ["auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score"]}})
    for _, r in targeted.iterrows():
        rows.append({"model": r["train_variant"], **{k: r.get(k, np.nan) for k in ["auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score"]}})
    out = pd.DataFrame(rows)
    write_csv(out, results_dir / "all_samples_model_comparison.csv")
    return out


def figure_index(diagnosis_plots: Path) -> None:
    entries = [
        ("results_source_matrix/plots/source_matrix_auroc_heatmap.png", "Cross-source AUROC matrix.", "Cross-source generalization", "Public tests are strong while all_samples is weak."),
        ("results_source_matrix/plots/source_matrix_auprc_heatmap.png", "Cross-source AUPRC matrix.", "Cross-source generalization", "AUPRC drops on all_samples across most training sources."),
        ("results_source_matrix/plots/mean_smd_by_test_set.png", "Mean SMD by test set.", "Distribution shift", "all_samples has the largest mean feature shift."),
        ("results_source_matrix/plots/all_samples_top_shifted_features.png", "Top all_samples shifted features.", "Distribution shift", "Scale-response and probability loss features drive the shift."),
        ("results_diagnosis/plots/all_samples_roc_curves_comparison.png", "ROC curves on all_samples.", "External performance", "M4-like training is the strongest among source models."),
        ("results_diagnosis/plots/all_samples_pr_curves_comparison.png", "PR curves on all_samples.", "External performance", "Precision-recall remains limited under shift."),
        ("results_ablation/combined_public_ablation_auroc_auprc.png", "Feature ablation performance.", "Feature contribution", "scale_response adds more signal than probability-only."),
        ("results_diagnosis/plots/feature_space_pca_by_source.png", "PCA by source.", "Feature-space shift", "all_samples separates from public tests in feature space."),
        ("results_targeted/plots/m4_targeted_performance.png", "M4-targeted variants.", "Targeted optimization", "Targeted sampling did not fully remove all_samples shift."),
    ]
    with open(diagnosis_plots / "FIGURE_INDEX.md", "w", encoding="utf-8") as handle:
        handle.write("# Figure Index\n\n")
        for path, desc, slide, takeaway in entries:
            handle.write(f"## {Path(path).name}\n\n")
            handle.write(f"- Path: `{path}`\n")
            handle.write(f"- Shows: {desc}\n")
            handle.write(f"- Suggested slide: {slide}\n")
            handle.write(f"- Takeaway: {takeaway}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    diagnosis_dir = ROOT / "results_diagnosis"
    diagnosis_plots = diagnosis_dir / "plots"
    source_plots = ROOT / "results_source_matrix" / "plots"
    targeted_dir = ROOT / "results_targeted"
    targeted_plots = targeted_dir / "plots"
    checkpoint_dir = ROOT / "checkpoints_targeted"
    for d in [diagnosis_dir, diagnosis_plots, source_plots, targeted_dir, targeted_plots, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    error_analysis(ROOT / args.external_features, ROOT / args.external_test, diagnosis_dir, diagnosis_plots)
    targeted = train_targeted_variants(ROOT / args.features, ROOT / args.source_splits, ROOT / args.external_features, ROOT / args.external_test, checkpoint_dir, targeted_dir, args.seed)
    model_comparison(targeted, targeted_dir)
    plot_heatmaps(ROOT / "results_source_matrix" / "source_generalization_matrix.csv", source_plots)
    plot_public_drop(ROOT / "results_source_matrix" / "source_generalization_matrix.csv", source_plots)
    plot_shift_figures(ROOT / "results_source_matrix" / "distribution_shift_report.csv", ROOT / "results_source_matrix" / "distribution_shift_summary.json", source_plots)
    plot_feature_distributions(ROOT / args.features, ROOT / args.external_features, ROOT / args.external_test, ROOT / args.source_splits, diagnosis_plots)
    plot_roc_pr_comparison(diagnosis_plots, targeted)
    plot_ablation_and_optimized()
    plot_targeted(targeted, targeted_plots)
    plot_feature_space(ROOT / args.features, ROOT / args.external_features, ROOT / args.external_test, ROOT / args.source_splits, diagnosis_plots, args.seed)
    figure_index(diagnosis_plots)
    save_json({"created_at": now(), "targeted_variants": targeted.to_dict(orient="records")}, diagnosis_dir / "diagnosis_targeted_manifest.json")
    print("Wrote diagnosis, targeted results, and presentation figures.")


if __name__ == "__main__":
    main()
