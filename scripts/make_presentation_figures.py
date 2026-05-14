#!/usr/bin/env python3
"""Create presentation-ready clean figures from existing experiment outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_source_matrix_eval import external_frame, make_xy, select_clean_columns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")

TEST_LABELS = {
    "all_samples": "all_samples",
    "ghostbuster_strict_test": "Ghostbuster",
    "hc3_plus_strict_test": "HC3+",
    "m4_strict_test": "M4",
}
TRAIN_LABELS = {
    "train_ghostbuster": "Ghostbuster",
    "train_m4": "M4",
    "train_hc3_plus": "HC3+",
    "train_combined_strict": "Combined",
    "train_balanced_combined_strict": "Balanced combined",
    "train_leave_out_ghostbuster": "M4+HC3",
    "train_leave_out_m4": "Ghostbuster+HC3",
    "train_leave_out_hc3_plus": "Ghostbuster+M4",
}
FEATURE_LABELS = {
    "scale_ppl_response_variance": "scale PPL variance",
    "scale_bottom_10_percent_loss_mean_response_variance": "scale bottom-10% loss variance",
    "qwen25_1_5b_loss_min": "Qwen1.5B min loss",
    "qwen25_1_5b_ppl": "Qwen1.5B PPL",
    "scale_ppl_abs_delta_7b_to_14b": "scale PPL delta 7B->14B",
    "scale_ppl_response_range": "scale PPL range",
    "scale_ppl_abs_delta_1_5b_to_7b": "scale PPL delta 1.5B->7B",
    "scale_ppl_global_intercept": "scale PPL intercept",
    "qwen25_7b_loss_min": "Qwen7B min loss",
    "scale_ppl_global_slope": "scale PPL slope",
    "scale_ppl_normalized_area": "scale PPL normalized area",
    "scale_ppl_response_area": "scale PPL area",
}
VARIANT_LABELS = {
    "m4_only_cleaned": "M4 only",
    "m4_generator_balanced": "M4 generator-balanced",
    "m4_plus_hc3": "M4+HC3",
    "m4_plus_hc3_without_extreme_domains": "M4+HC3 no extreme domains",
    "m4_label_balanced": "M4 label-balanced",
}


def warn(msg: str) -> None:
    warnings.warn(msg)


def save_fig(fig: plt.Figure, clean_dir: Path, stem: str, dpi: int, made: list[str]) -> None:
    png = clean_dir / f"{stem}.png"
    pdf = clean_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    made.extend([str(png), str(pdf)])


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"missing input: {path}")
        return None
    return pd.read_csv(path)


def load_split_features(source_splits: Path, features: Path, clean_cols: list[str]) -> dict[str, pd.DataFrame]:
    feature_df = pd.read_csv(features)
    feature_by_id = feature_df[["id", *clean_cols]].copy()
    frames: dict[str, pd.DataFrame] = {}
    for source in ["ghostbuster", "m4", "hc3_plus"]:
        for split in ["train", "dev", "test"]:
            meta = pd.read_csv(source_splits / f"{source}_strict_{split}.csv")
            meta["label"] = pd.to_numeric(meta["label"], errors="coerce").astype(int)
            frames[f"{source}_{split}"] = meta.merge(feature_by_id, on="id", how="left", validate="one_to_one")
    return frames


def annotate_bars(ax, fmt="{:.2f}", fontsize=10) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append("" if np.isnan(h) else fmt.format(h))
        ax.bar_label(container, labels=labels, padding=2, fontsize=fontsize)


def source_heatmap(metric: str, title: str, stem: str, clean_dir: Path, dpi: int, made: list[str], skipped: list[str]) -> None:
    df = read_csv(ROOT / "results_source_matrix" / "source_generalization_matrix.csv")
    if df is None:
        skipped.append(stem)
        return
    rows = list(TRAIN_LABELS)
    cols = ["all_samples", "ghostbuster_strict_test", "hc3_plus_strict_test", "m4_strict_test"]
    sub = df[df["train_name"].isin(rows) & df["test_name"].isin(cols)].copy()
    pivot = sub.pivot(index="train_name", columns="test_name", values=metric).reindex(index=rows, columns=cols)
    pivot.index = [TRAIN_LABELS[x] for x in pivot.index]
    pivot.columns = [TEST_LABELS[x] for x in pivot.columns]
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": metric.upper()}, ax=ax, linewidths=0.6, linecolor="white")
    ax.set_title(title, pad=14, fontsize=18)
    ax.set_xlabel("Test set", fontsize=14)
    ax.set_ylabel("Train set", fontsize=14)
    ax.add_patch(plt.Rectangle((0, 0), 1, len(pivot.index), fill=False, edgecolor="red", linewidth=3))
    same = [("Ghostbuster", "Ghostbuster"), ("M4", "M4"), ("HC3+", "HC3+")]
    for train, test in same:
        if train in pivot.index and test in pivot.columns:
            i = list(pivot.index).index(train)
            j = list(pivot.columns).index(test)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="white", linewidth=3))
    save_fig(fig, clean_dir, stem, dpi, made)


def mean_smd(clean_dir: Path, dpi: int, made: list[str], skipped: list[str]) -> None:
    path = ROOT / "results_source_matrix" / "distribution_shift_summary.json"
    if not path.exists():
        skipped.append("slide03_mean_smd_by_test_set")
        warn(f"missing input: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))["test_sets"]
    order = ["ghostbuster_test", "m4_test", "hc3_plus_test", "all_samples"]
    labels = ["Ghostbuster", "M4", "HC3+", "all_samples"]
    vals = [data[k]["mean_smd"] for k in order]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(labels, vals, color=sns.color_palette("deep", 4), edgecolor="black", linewidth=[1, 1, 1, 2])
    bars[-1].set_hatch("//")
    ax.set_title("all_samples is substantially shifted in feature space", fontsize=16, pad=12)
    ax.set_ylabel("Mean SMD")
    ax.set_xlabel("Test set")
    ax.set_ylim(0, max(vals) * 1.22)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    save_fig(fig, clean_dir, "slide03_mean_smd_by_test_set", dpi, made)


def top_shifted(clean_dir: Path, dpi: int, made: list[str], skipped: list[str]) -> None:
    df = read_csv(ROOT / "results_source_matrix" / "distribution_shift_report.csv")
    if df is None:
        skipped.append("slide04_top_shifted_features_all_samples")
        return
    top = df[df["test_name"].eq("all_samples")].sort_values("standardized_mean_difference", ascending=False).head(12).copy()
    top["label"] = top["feature"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ")))
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.barplot(data=top.iloc[::-1], x="standardized_mean_difference", y="label", color="#4C78A8", ax=ax)
    ax.set_title("Largest shifts occur in probability and scale-response features", fontsize=16, pad=12)
    ax.set_xlabel("Standardized mean difference")
    ax.set_ylabel("Feature")
    save_fig(fig, clean_dir, "slide04_top_shifted_features_all_samples", dpi, made)


def pca_source(clean_dir: Path, dpi: int, made: list[str], skipped: list[str], seed: int = 42) -> None:
    paths = [
        ROOT / "features_by_dataset" / "combined_public_full_allfeatures" / "all_features.csv",
        ROOT / "features_external" / "all_samples_full_allfeatures" / "all_features.csv",
        ROOT / "data" / "source_splits",
        ROOT / "data" / "test" / "all_samples_prepared.csv",
    ]
    if any(not p.exists() for p in paths):
        skipped.append("slide05_pca_feature_space_by_source")
        warn("missing PCA input")
        return
    feature_df = pd.read_csv(paths[0])
    clean_cols, _ = select_clean_columns(feature_df)
    frames = load_split_features(paths[2], paths[0], clean_cols)
    all_samples = external_frame(paths[1], paths[3], clean_cols)
    parts = []
    for key, label in [("ghostbuster_test", "Ghostbuster"), ("m4_test", "M4"), ("hc3_plus_test", "HC3+")]:
        tmp = frames[key].copy()
        tmp["plot_source"] = label
        parts.append(tmp)
    tmp = all_samples.copy()
    tmp["plot_source"] = "all_samples"
    parts.append(tmp)
    df = pd.concat(parts, ignore_index=True)
    df = df.groupby("plot_source", group_keys=False).apply(lambda g: g if g.name == "all_samples" else g.sample(n=min(len(g), 1000), random_state=seed)).reset_index(drop=True)
    x, _, _ = make_xy(df, clean_cols)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(StandardScaler().fit_transform(x))
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    palette = {"all_samples": "#E45756", "Ghostbuster": "#4C78A8", "M4": "#54A24B", "HC3+": "#F58518"}
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="plot_source", hue_order=["all_samples", "Ghostbuster", "M4", "HC3+"], palette=palette, alpha=0.72, s=32, ax=ax)
    ax.set_title("all_samples separates from public test splits in feature space", fontsize=15, pad=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(title="Source", frameon=True)
    save_fig(fig, clean_dir, "slide05_pca_feature_space_by_source", dpi, made)


def ghostbuster_reversal(clean_dir: Path, dpi: int, made: list[str], skipped: list[str]) -> None:
    pred_path = ROOT / "results_source_matrix" / "train_ghostbuster_to_all_samples" / "predictions.csv"
    if not pred_path.exists():
        candidates = list((ROOT / "results_source_matrix").glob("*ghostbuster*all_samples*/predictions.csv"))
        pred_path = candidates[0] if candidates else pred_path
    pred = read_csv(pred_path)
    if pred is None:
        skipped.append("slide06_ghostbuster_probability_reversal")
        return
    metrics = read_csv(pred_path.parent / "metrics.csv")
    met = metrics.iloc[0].to_dict() if metrics is not None and not metrics.empty else {}
    pred["label_name"] = pd.to_numeric(pred["label"], errors="coerce").astype(int).map({0: "Human", 1: "AI"})
    human_mean = pred.loc[pred["label_name"].eq("Human"), "ai_probability"].mean()
    ai_mean = pred.loc[pred["label_name"].eq("AI"), "ai_probability"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    sns.histplot(data=pred, x="ai_probability", hue="label_name", bins=24, stat="density", common_norm=False, ax=axes[0])
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.4)
    axes[0].set_title("Histogram")
    sns.kdeplot(data=pred, x="ai_probability", hue="label_name", common_norm=False, ax=axes[1], warn_singular=False)
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.4)
    axes[1].set_title("Density")
    sns.violinplot(data=pred, x="label_name", y="ai_probability", ax=axes[2], cut=0)
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1.4)
    axes[2].set_title("Violin")
    fig.suptitle(
        f"Ghostbuster-trained detector reverses on all_samples\n"
        f"AUROC={met.get('auroc', np.nan):.3f}, AUPRC={met.get('auprc', np.nan):.3f}, F1={met.get('f1', np.nan):.3f}; "
        f"Human mean AI probability={human_mean:.3f}, AI mean AI probability={ai_mean:.3f}",
        fontsize=14,
    )
    save_fig(fig, clean_dir, "slide06_ghostbuster_probability_reversal", dpi, made)


def grouped_bar(input_path: Path, id_col: str, label_map: dict[str, str], keep_ids: list[str] | None, metrics: list[str], title: str, stem: str, clean_dir: Path, dpi: int, made: list[str], skipped: list[str], ylim: tuple[float, float] | None = None) -> None:
    df = read_csv(input_path)
    if df is None:
        skipped.append(stem)
        return
    if keep_ids is not None:
        df = df[df[id_col].isin(keep_ids)].copy()
    df["label"] = df[id_col].map(lambda x: label_map.get(x, x))
    long = df[["label", *metrics]].melt(id_vars="label", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    sns.barplot(data=long, x="label", y="value", hue="metric", ax=ax)
    annotate_bars(ax, fmt="{:.3f}", fontsize=9)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(metrics), frameon=True)
    save_fig(fig, clean_dir, stem, dpi, made)


def roc_pr(clean_dir: Path, dpi: int, made: list[str], skipped: list[str]) -> None:
    model_dirs = {
        "Ghostbuster": ROOT / "results_source_matrix" / "train_ghostbuster_to_all_samples",
        "M4": ROOT / "results_source_matrix" / "train_m4_to_all_samples",
        "HC3+": ROOT / "results_source_matrix" / "train_hc3_plus_to_all_samples",
        "Combined": ROOT / "results_source_matrix" / "train_combined_strict_to_all_samples",
        "M4+HC3": ROOT / "results_source_matrix" / "train_leave_out_ghostbuster_to_all_samples",
    }
    fig_roc, ax_roc = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    fig_pr, ax_pr = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    baseline = None
    plotted = 0
    for label, d in model_dirs.items():
        pred = read_csv(d / "predictions.csv")
        if pred is None:
            continue
        y = pd.to_numeric(pred["label"], errors="coerce").astype(int)
        p = pd.to_numeric(pred["ai_probability"], errors="coerce")
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        auroc = auc(fpr, tpr)
        auprc = auc(rec, prec)
        baseline = y.mean()
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{label} ({auroc:.3f})")
        ax_pr.plot(rec, prec, linewidth=2, label=f"{label} ({auprc:.3f})")
        plotted += 1
    if not plotted:
        skipped.extend(["slide09_all_samples_roc_comparison", "slide10_all_samples_pr_comparison"])
        plt.close(fig_roc)
        plt.close(fig_pr)
        return
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_title("ROC comparison on all_samples", fontsize=16, pad=12)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(title="Model (AUROC)", fontsize=9)
    if baseline is not None:
        ax_pr.axhline(baseline, color="black", linestyle="--", label=f"Random ({baseline:.2f})")
    ax_pr.set_title("Precision-recall comparison on all_samples", fontsize=16, pad=12)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(title="Model (AUPRC)", fontsize=9)
    save_fig(fig_roc, clean_dir, "slide09_all_samples_roc_comparison", dpi, made)
    save_fig(fig_pr, clean_dir, "slide10_all_samples_pr_comparison", dpi, made)


def copy_originals(original_dir: Path) -> tuple[list[str], list[str]]:
    rels = [
        "results_source_matrix/plots/source_matrix_auroc_heatmap.png",
        "results_source_matrix/plots/source_matrix_auroc_heatmap.pdf",
        "results_source_matrix/plots/source_matrix_auprc_heatmap.png",
        "results_source_matrix/plots/source_matrix_auprc_heatmap.pdf",
        "results_source_matrix/plots/mean_smd_by_test_set.png",
        "results_source_matrix/plots/mean_smd_by_test_set.pdf",
        "results_source_matrix/plots/all_samples_top_shifted_features.png",
        "results_source_matrix/plots/all_samples_top_shifted_features.pdf",
        "results_diagnosis/plots/feature_space_pca_by_source.png",
        "results_diagnosis/plots/feature_space_pca_by_source.pdf",
        "results_diagnosis/plots/all_samples_probability_distribution_ghostbuster.png",
        "results_diagnosis/plots/all_samples_probability_distribution_ghostbuster.pdf",
        "results_ablation/combined_public_ablation_auroc_auprc.png",
        "results_ablation/combined_public_ablation_auroc_auprc.pdf",
        "results_targeted/plots/m4_targeted_performance.png",
        "results_targeted/plots/m4_targeted_performance.pdf",
    ]
    copied, missing = [], []
    original_dir.mkdir(parents=True, exist_ok=True)
    for rel in rels:
        src = ROOT / rel
        if src.exists():
            dst = original_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
        else:
            warn(f"missing original figure: {src}")
            missing.append(rel)
    return copied, missing


def write_docs(output_dir: Path, made: list[str], skipped: list[str]) -> None:
    figs = [
        ("slide01_source_matrix_auroc_heatmap.png", "results_source_matrix/source_generalization_matrix.csv", "AUROC source generalization heatmap.", "Models perform well on public splits but drop sharply on all_samples.", "1"),
        ("slide02_source_matrix_auprc_heatmap.png", "results_source_matrix/source_generalization_matrix.csv", "AUPRC source generalization heatmap.", "AUPRC confirms the all_samples external gap.", "1"),
        ("slide03_mean_smd_by_test_set.png", "results_source_matrix/distribution_shift_summary.json", "Mean SMD by test set.", "all_samples has much larger mean SMD than public test splits.", "2"),
        ("slide04_top_shifted_features_all_samples.png", "results_source_matrix/distribution_shift_report.csv", "Top shifted all_samples features.", "The largest shifts occur in Qwen probability and scale-response features.", "3"),
        ("slide05_pca_feature_space_by_source.png", "existing cleaned full_allfeatures and strict test splits", "PCA feature-space view by source.", "all_samples separates from public test splits.", "2"),
        ("slide06_ghostbuster_probability_reversal.png", "results_source_matrix/train_ghostbuster_to_all_samples/predictions.csv", "Ghostbuster probability distributions on all_samples.", "Ghostbuster assigns higher AI probability to human than AI text.", "4"),
        ("slide07_ablation_scale_response_contribution.png", "results_ablation/combined_public_feature_ablation_summary.csv", "Feature ablation AUROC/AUPRC.", "Basic+Scale outperforms Basic+Probability.", "5"),
        ("slide08_m4_targeted_variants.png", "results_targeted/m4_targeted_summary.csv", "M4-targeted variant performance.", "M4-only best matches all_samples but remains imperfect.", "6"),
        ("slide09_all_samples_roc_comparison.png", "results_source_matrix/*_to_all_samples/predictions.csv", "ROC curves on all_samples.", "M4-like models are strongest but still limited.", "6"),
        ("slide10_all_samples_pr_comparison.png", "results_source_matrix/*_to_all_samples/predictions.csv", "PR curves on all_samples.", "PR performance remains modest under shift.", "6"),
    ]
    with open(output_dir / "FIGURE_INDEX.md", "w", encoding="utf-8") as handle:
        handle.write("# Presentation Figure Index\n\n")
        for name, src, shows, takeaway, slide in figs:
            status = "generated" if str(output_dir / "figures_clean" / name) in made or (output_dir / "figures_clean" / name).exists() else "skipped"
            handle.write(f"## {name}\n\n")
            handle.write(f"- Status: {status}\n")
            handle.write(f"- Data source: `{src}`\n")
            handle.write(f"- Shows: {shows}\n")
            handle.write(f"- Takeaway: {takeaway}\n")
            handle.write(f"- Suggested PPT page: Slide {slide}\n\n")
        if skipped:
            handle.write("## Skipped\n\n")
            for item in skipped:
                handle.write(f"- {item}\n")
    guide = """# Slide Guide

Slide 1:
Title: Public benchmarks show strong performance, but all_samples exposes a gap
Figure: slide01_source_matrix_auroc_heatmap.png
Takeaway: Models perform well on public splits but drop sharply on all_samples.

Slide 2:
Title: all_samples is a shifted external set
Figure: slide03_mean_smd_by_test_set.png
Takeaway: all_samples has much larger mean SMD than public test splits.

Slide 3:
Title: Shift concentrates in probability and scale-response features
Figure: slide04_top_shifted_features_all_samples.png
Takeaway: The largest shifts occur in Qwen probability and scale-response features.

Slide 4:
Title: Ghostbuster artifacts reverse on all_samples
Figure: slide06_ghostbuster_probability_reversal.png
Takeaway: Ghostbuster assigns higher AI probability to human text than AI text on all_samples.

Slide 5:
Title: Scale-response provides useful signal
Figure: slide07_ablation_scale_response_contribution.png
Takeaway: Basic+Scale outperforms Basic+Probability, supporting the value of scale-response profiling.

Slide 6:
Title: M4 is the closest public source, but low-FPR detection remains weak
Figure: slide08_m4_targeted_variants.png
Takeaway: M4-only performs best among public-source variants, but detection remains imperfect.
"""
    (output_dir / "SLIDE_GUIDE.md").write_text(guide, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results_presentation")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    clean_dir = output_dir / "figures_clean"
    original_dir = output_dir / "figures_original"
    clean_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    skipped: list[str] = []

    source_heatmap("auroc", "External all_samples shows a large generalization gap", "slide01_source_matrix_auroc_heatmap", clean_dir, args.dpi, made, skipped)
    source_heatmap("auprc", "AUPRC confirms the same external generalization gap", "slide02_source_matrix_auprc_heatmap", clean_dir, args.dpi, made, skipped)
    mean_smd(clean_dir, args.dpi, made, skipped)
    top_shifted(clean_dir, args.dpi, made, skipped)
    pca_source(clean_dir, args.dpi, made, skipped)
    ghostbuster_reversal(clean_dir, args.dpi, made, skipped)
    grouped_bar(
        ROOT / "results_ablation" / "combined_public_feature_ablation_summary.csv",
        "feature_set",
        {
            "basic_only": "Basic",
            "probability_only": "Prob. only",
            "scale_response_only": "Scale only",
            "basic_plus_probability": "Basic+Prob.",
            "basic_plus_scale_response": "Basic+Scale",
            "full_cleaned": "Full",
        },
        ["basic_only", "probability_only", "scale_response_only", "basic_plus_probability", "basic_plus_scale_response", "full_cleaned"],
        ["auroc", "auprc"],
        "Scale-response features add transferable signal",
        "slide07_ablation_scale_response_contribution",
        clean_dir,
        args.dpi,
        made,
        skipped,
        ylim=(0.45, 0.68),
    )
    grouped_bar(
        ROOT / "results_targeted" / "m4_targeted_summary.csv",
        "train_variant",
        VARIANT_LABELS,
        ["m4_only_cleaned", "m4_generator_balanced", "m4_plus_hc3", "m4_plus_hc3_without_extreme_domains", "m4_label_balanced"],
        ["auroc", "auprc", "f1"],
        "M4-based training best matches all_samples, but remains imperfect",
        "slide08_m4_targeted_variants",
        clean_dir,
        args.dpi,
        made,
        skipped,
        ylim=(0.45, 0.72),
    )
    roc_pr(clean_dir, args.dpi, made, skipped)
    copied, missing_originals = copy_originals(original_dir)
    skipped.extend([f"original:{x}" for x in missing_originals])
    write_docs(output_dir, made, skipped)
    save_json = {
        "clean_files": made,
        "copied_original_files": copied,
        "skipped": skipped,
        "clean_png_count": len(list(clean_dir.glob("*.png"))),
        "clean_pdf_count": len(list(clean_dir.glob("*.pdf"))),
        "original_file_count": len(copied),
    }
    (output_dir / "presentation_figure_manifest.json").write_text(json.dumps(save_json, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(save_json, indent=2))


if __name__ == "__main__":
    main()
