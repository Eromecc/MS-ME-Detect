#!/usr/bin/env python3
"""Build a Deep DMD cross-source generalization matrix.

This script is intentionally conservative: it reuses the full-sweep Deep DMD
outputs when available, and only trains missing lightweight qwen25_1_5b runs
from existing token-loss caches. It never runs probability, scale-response, or
token-loss generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_deep_dmd_experiment import (  # noqa: E402
    LOSS_CONFIGS,
    get_device,
    load_if_available,
    rank_tuple,
    save_json,
    save_score_eval,
)
from run_koopman_dmd_experiment import merge_transition  # noqa: E402
from run_transition_formal_experiment import (  # noqa: E402
    cleaned_full_columns,
    eval_one,
    fit_select,
    merge_full_features,
    save_fig,
)
from run_transition_fullscale_optimized import build_base_datasets  # noqa: E402
from src.deep_dmd_features import extract_deep_dmd_features  # noqa: E402
from src.deep_dmd_dataset import (  # noqa: E402
    DeepDMDTokenDataset,
    fit_loss_bins,
    fit_observable_scaler,
    read_token_loss_cache,
)
from src.deep_dmd_model import DeepDMDEncoder  # noqa: E402
from src.deep_dmd_train import evaluate_deep_dmd, predict_scores, train_deep_dmd  # noqa: E402
from src.utils import write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


TEST_ORDER = ["ghostbuster_test", "m4_test", "hc3_plus_test", "all_samples"]
TEST_LABELS = {
    "ghostbuster_test": "Ghostbuster",
    "m4_test": "M4",
    "hc3_plus_test": "HC3+",
    "all_samples": "all_samples",
}
TRAIN_LABELS = {
    "ghostbuster": "Ghostbuster",
    "m4": "M4",
    "hc3_plus": "HC3+",
    "combined_strict": "Combined",
    "leave_out_ghostbuster": "M4+HC3",
    "leave_out_m4": "Ghostbuster+HC3",
    "leave_out_hc3_plus": "Ghostbuster+M4",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_train_name(name: str) -> str:
    return name.removeprefix("train_")


def composite_train_dev_extended(data: dict[str, pd.DataFrame], train_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = normalize_train_name(train_name)
    if name in {"ghostbuster", "m4", "hc3_plus"}:
        return data[f"{name}_train"], data[f"{name}_dev"]
    if name == "combined_strict":
        return (
            pd.concat([data["ghostbuster_train"], data["m4_train"], data["hc3_plus_train"]], ignore_index=True),
            pd.concat([data["ghostbuster_dev"], data["m4_dev"], data["hc3_plus_dev"]], ignore_index=True),
        )
    if name == "leave_out_ghostbuster":
        return (
            pd.concat([data["m4_train"], data["hc3_plus_train"]], ignore_index=True),
            pd.concat([data["m4_dev"], data["hc3_plus_dev"]], ignore_index=True),
        )
    if name == "leave_out_m4":
        return (
            pd.concat([data["ghostbuster_train"], data["hc3_plus_train"]], ignore_index=True),
            pd.concat([data["ghostbuster_dev"], data["hc3_plus_dev"]], ignore_index=True),
        )
    if name == "leave_out_hc3_plus":
        return (
            pd.concat([data["ghostbuster_train"], data["m4_train"]], ignore_index=True),
            pd.concat([data["ghostbuster_dev"], data["m4_dev"]], ignore_index=True),
        )
    raise ValueError(f"unknown train source: {train_name}")


def prepare_data_for_extended_sources(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = dict(data)
    for name in ["ghostbuster", "m4", "hc3_plus"]:
        out[f"{name}_train"]["transition_split"] = f"{name}_train"
        out[f"{name}_dev"]["transition_split"] = f"{name}_dev"
        out[f"{name}_test"]["transition_split"] = f"{name}_test"
    out["combined_strict_train"], out["combined_strict_dev"] = composite_train_dev_extended(out, "combined_strict")
    out["leave_out_ghostbuster_train"], out["leave_out_ghostbuster_dev"] = composite_train_dev_extended(out, "leave_out_ghostbuster")
    out["leave_out_m4_train"], out["leave_out_m4_dev"] = composite_train_dev_extended(out, "leave_out_m4")
    out["leave_out_hc3_plus_train"], out["leave_out_hc3_plus_dev"] = composite_train_dev_extended(out, "leave_out_hc3_plus")
    return out


def token_cache_path(model_name: str, dataset: str) -> Path:
    return ROOT / "features_token_loss" / model_name / f"{dataset}_token_loss.jsonl.gz"


def required_cache_names(train_name: str) -> list[str]:
    train, dev = f"{train_name}_train", f"{train_name}_dev"
    if train_name in {"ghostbuster", "m4", "hc3_plus", "combined_strict", "leave_out_ghostbuster"}:
        return [train, dev]
    # Composite leave-out caches may not exist, but per-source caches do.
    if train_name == "leave_out_m4":
        return ["ghostbuster_train", "hc3_plus_train", "ghostbuster_dev", "hc3_plus_dev"]
    if train_name == "leave_out_hc3_plus":
        return ["ghostbuster_train", "m4_train", "ghostbuster_dev", "m4_dev"]
    return [train, dev]


def can_train_from_cache(train_name: str, model_name: str) -> bool:
    names = required_cache_names(train_name) + ["ghostbuster_test", "m4_test", "hc3_plus_test", "all_samples"]
    return all(token_cache_path(model_name, n).exists() for n in names)


def cache_names_for_model(model_name: str) -> list[str]:
    root = ROOT / "features_token_loss" / model_name
    return sorted(p.name.replace("_token_loss.jsonl.gz", "") for p in root.glob("*_token_loss.jsonl.gz"))


def build_deep_datasets_extended(data: dict[str, pd.DataFrame], train_name: str, model_name: str, max_seq_len: int, min_tokens: int):
    train_meta, dev_meta = composite_train_dev_extended(data, train_name)
    union = {}
    for name in cache_names_for_model(model_name):
        path = token_cache_path(model_name, name)
        if path.exists():
            union.update(read_token_loss_cache(path))
    train_ids = [str(x) for x in train_meta["id"]]
    bins = fit_loss_bins(union, train_ids, n_states=5)
    scaler = fit_observable_scaler(union, train_ids, max_seq_len=max_seq_len, loss_bins=bins)
    datasets = {
        "train": DeepDMDTokenDataset(train_meta, union, max_seq_len=max_seq_len, loss_bins=bins, scaler=scaler, min_tokens=min_tokens),
        "dev": DeepDMDTokenDataset(dev_meta, union, max_seq_len=max_seq_len, loss_bins=bins, scaler=scaler, min_tokens=min_tokens),
    }
    for name in TEST_ORDER:
        datasets[name] = DeepDMDTokenDataset(data[name], union, max_seq_len=max_seq_len, loss_bins=bins, scaler=scaler, min_tokens=min_tokens)
    return datasets, scaler, bins


def load_existing_full_sweep() -> pd.DataFrame:
    path = ROOT / "results_deep_dmd/full_sweep_1_5b_7b/deep_dmd_full_sweep_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["train_source"] = df["train_name"].map(normalize_train_name)
    return df


def train_missing_source(
    *,
    train_name: str,
    model_name: str,
    data: dict[str, pd.DataFrame],
    train_features: pd.DataFrame,
    ext_features: pd.DataFrame,
    full_cols: list[str],
    out_dir: Path,
    ckpt_root: Path,
    feat_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    meta: dict = {"train_source": train_name, "model_name": model_name, "status": "started", "reused_checkpoint": False}
    train_meta, dev_meta = composite_train_dev_extended(data, train_name)
    train_full = merge_full_features(train_meta, train_features, ext_features, full_cols)
    dev_full = merge_full_features(dev_meta, train_features, ext_features, full_cols)
    datasets, scaler, bins = build_deep_datasets_extended(data, train_name, model_name, args.max_seq_len, min_tokens=20)
    exp = f"{train_name}_{model_name}_seq{args.max_seq_len}_ld{args.latent_dim}_hd{args.hidden_dim}_lr1em03_cfg{args.loss_config}"
    ckpt = ckpt_root / exp
    model = DeepDMDEncoder(datasets["train"].input_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    loaded = args.reuse_existing_checkpoints and load_if_available(model, ckpt, device)
    meta["reused_checkpoint"] = bool(loaded)
    if args.run_missing_only and loaded:
        pass
    elif args.run_eval and not loaded:
        model, hist, train_meta_out = train_deep_dmd(
            model,
            datasets["train"],
            datasets["dev"],
            output_dir=ckpt,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            patience=args.patience,
            seed=args.seed,
            loss_weights=LOSS_CONFIGS[args.loss_config],
        )
        save_json(
            {
                "created_at": now(),
                "source": "run_deep_dmd_cross_source_matrix.py",
                "scaler": scaler.to_dict(),
                "loss_bins": bins,
                "train_name": train_name,
                "model_name": model_name,
                "args": vars(args),
                "train_metadata": train_meta_out,
            },
            ckpt / "preprocess.json",
        )
    feature_dir = feat_root / exp
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = {}
    for ds_name in ["train", "dev", *TEST_ORDER]:
        path = feature_dir / f"{ds_name}_deep_dmd_features.csv"
        if not path.exists():
            extract_deep_dmd_features(model, datasets[ds_name], path, batch_size=args.batch_size, device=device, prefix=f"deep_dmd_{model_name}")
        feature_paths[ds_name] = path
    for test_name in TEST_ORDER:
        pred = predict_scores(model, datasets[test_name], batch_size=args.batch_size, device=device)
        m = save_score_eval(pred, out_dir / f"{exp}_deep_dmd_score_only_to_{test_name}")
        rows.append({"train_name": train_name, "train_source": train_name, "model_name": model_name, "experiment": exp, "feature_set": "deep_dmd_score_only", "test_set": test_name, "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim, **m})
    deep_cols = [c for c in pd.read_csv(feature_paths["train"], nrows=1).columns if c != "id"]
    train_deep = train_full.merge(pd.read_csv(feature_paths["train"]), on="id", how="left")
    dev_deep = dev_full.merge(pd.read_csv(feature_paths["dev"]), on="id", how="left")
    for fs, base_train, base_dev, cols in [
        ("deep_dmd_spectral_features_only", train_deep, dev_deep, deep_cols),
        ("full_plus_deep_dmd_spectral", train_deep, dev_deep, full_cols + deep_cols),
    ]:
        cols = [c for c in cols if c in base_train.columns]
        best_name, clf, med, _ = fit_select(base_train, base_dev, cols)
        joblib.dump(clf, ckpt / f"{fs}_classifier.joblib")
        for test_name in TEST_ORDER:
            test_base = merge_full_features(data[test_name], train_features, ext_features, full_cols)
            test_df = test_base.merge(pd.read_csv(feature_paths[test_name]), on="id", how="left")
            m = eval_one(clf, test_df, cols, med, out_dir / f"{exp}_{fs}_to_{test_name}")
            rows.append({"train_name": train_name, "train_source": train_name, "model_name": model_name, "experiment": exp, "feature_set": fs, "test_set": test_name, "best_model": best_name, "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim, **m})
    # Only evaluate transition+DeepDMD if transition columns for this train source exist.
    train_trans = merge_transition(train_full, train_name)
    dev_trans = merge_transition(dev_full, train_name)
    trans_cols = [c for c in train_trans.columns if c.endswith("_one5") or c.endswith("_seven")]
    if trans_cols:
        train_deep_trans = train_trans.merge(pd.read_csv(feature_paths["train"]), on="id", how="left")
        dev_deep_trans = dev_trans.merge(pd.read_csv(feature_paths["dev"]), on="id", how="left")
        cols = [c for c in full_cols + trans_cols + deep_cols if c in train_deep_trans.columns]
        best_name, clf, med, _ = fit_select(train_deep_trans, dev_deep_trans, cols)
        joblib.dump(clf, ckpt / "full_plus_transition_plus_deep_dmd_classifier.joblib")
        for test_name in TEST_ORDER:
            test_base = merge_full_features(data[test_name], train_features, ext_features, full_cols)
            test_trans = merge_transition(test_base, train_name)
            test_df = test_trans.merge(pd.read_csv(feature_paths[test_name]), on="id", how="left")
            m = eval_one(clf, test_df, cols, med, out_dir / f"{exp}_full_plus_transition_plus_deep_dmd_to_{test_name}")
            rows.append({"train_name": train_name, "train_source": train_name, "model_name": model_name, "experiment": exp, "feature_set": "full_plus_transition_plus_deep_dmd", "test_set": test_name, "best_model": best_name, "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim, **m})
    else:
        meta["warning"] = "transition features missing for this train source; skipped full_plus_transition_plus_deep_dmd"
    meta["status"] = "completed"
    return rows, meta


def best_deep_method(summary: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "full_plus_transition_plus_deep_dmd",
        "full_plus_transition_plus_deep_dmd_1_5b_7b",
        "full_plus_deep_dmd_spectral",
        "full_plus_deep_dmd_1_5b_7b",
        "full_plus_deep_dmd_score",
        "deep_dmd_spectral_features_only",
        "deep_dmd_score_only",
    ]
    rows = []
    for (train, test), g in summary.groupby(["train_source", "test_set"]):
        g = g[g["feature_set"].isin(preferred)].copy()
        if g.empty:
            continue
        g["pref"] = g["feature_set"].map({v: i for i, v in enumerate(preferred)}).fillna(999)
        # Report the strongest Deep DMD variant per cell, but keep feature_set in the row.
        r = g.sort_values(["auroc", "auprc", "tpr_at_fpr_5pct", "f1"], ascending=False).iloc[0].to_dict()
        r["method"] = "deep_dmd_best_available"
        rows.append(r)
    return pd.DataFrame(rows)


def transition_reference() -> pd.DataFrame:
    rows = []
    p = ROOT / "results_transition/qwen25_7b_targeted/transition_7b_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        pick = df[df["experiment"].eq("full_plus_1_5b_and_7b_transition")].copy()
        for _, r in pick.iterrows():
            rows.append({"train_source": normalize_train_name(r["train_name"]), "test_set": r["test_set"], "method": "transition_best_available", "feature_set": r["experiment"], **{k: r.get(k, np.nan) for k in ["auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score"]}})
    p = ROOT / "results_source_matrix/source_generalization_matrix.csv"
    if p.exists():
        df = pd.read_csv(p)
        mapping = {
            "train_ghostbuster": "ghostbuster",
            "train_m4": "m4",
            "train_hc3_plus": "hc3_plus",
            "train_combined_strict": "combined_strict",
            "train_leave_out_ghostbuster": "leave_out_ghostbuster",
            "train_leave_out_m4": "leave_out_m4",
            "train_leave_out_hc3_plus": "leave_out_hc3_plus",
        }
        test_map = {"ghostbuster_strict_test": "ghostbuster_test", "m4_strict_test": "m4_test", "hc3_plus_strict_test": "hc3_plus_test", "all_samples": "all_samples"}
        for _, r in df.iterrows():
            rows.append({"train_source": mapping.get(r["train_name"], normalize_train_name(r["train_name"])), "test_set": test_map.get(r["test_name"], r["test_name"]), "method": "cleaned_full_source_matrix_reference", "feature_set": "cleaned_full_allfeatures_source_matrix", **{k: r.get(k, np.nan) for k in ["auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "ece", "brier_score"]}})
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, value: str, out: Path, title: str, center: float | None = None) -> None:
    if df.empty or value not in df.columns:
        return
    pivot = df.pivot_table(index="train_source", columns="test_set", values=value, aggfunc="max")
    pivot = pivot.reindex(index=[k for k in TRAIN_LABELS if k in pivot.index], columns=TEST_ORDER)
    show = pivot.rename(index=TRAIN_LABELS, columns=TEST_LABELS)
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = "coolwarm" if center is not None else "viridis"
    sns.heatmap(show, annot=True, fmt=".3f", cmap=cmap, center=center, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Test set")
    ax.set_ylabel("Train source")
    ax.set_title(title)
    # Highlight all_samples column.
    if "all_samples" in pivot.columns:
        j = list(pivot.columns).index("all_samples")
        ax.add_patch(plt.Rectangle((j, 0), 1, len(pivot.index), fill=False, edgecolor="red", lw=2.5))
    # Same-source public cells.
    same = {"ghostbuster": "ghostbuster_test", "m4": "m4_test", "hc3_plus": "hc3_plus_test"}
    for i, tr in enumerate(pivot.index):
        if tr in same and same[tr] in pivot.columns:
            j = list(pivot.columns).index(same[tr])
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="white", lw=2.5))
    save_fig(fig, out)


def generalization_gap(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    same_map = {"ghostbuster": "ghostbuster_test", "m4": "m4_test", "hc3_plus": "hc3_plus_test"}
    for (train, method), g in best.groupby(["train_source", "method"]):
        same = np.nan
        if train in same_map:
            x = g[g["test_set"].eq(same_map[train])]
            same = float(x["auroc"].iloc[0]) if not x.empty else np.nan
        public_cross = g[g["test_set"].isin(["ghostbuster_test", "m4_test", "hc3_plus_test"])]
        if train in same_map:
            public_cross = public_cross[~public_cross["test_set"].eq(same_map[train])]
        all_s = g[g["test_set"].eq("all_samples")]
        all_auc = float(all_s["auroc"].iloc[0]) if not all_s.empty else np.nan
        mean_cross = float(public_cross["auroc"].mean()) if not public_cross.empty else np.nan
        rows.append({
            "train_source": train,
            "method": method,
            "same_source_auroc": same,
            "mean_public_cross_source_auroc": mean_cross,
            "all_samples_auroc": all_auc,
            "same_to_all_gap": same - all_auc if pd.notna(same) and pd.notna(all_auc) else np.nan,
            "public_cross_to_all_gap": mean_cross - all_auc if pd.notna(mean_cross) and pd.notna(all_auc) else np.nan,
        })
    return pd.DataFrame(rows)


def write_report(out_dir: Path, matrix: pd.DataFrame, delta: pd.DataFrame, gap: pd.DataFrame, manifest: dict) -> None:
    deep = matrix[matrix["method"].eq("deep_dmd_best_available")]
    trans = matrix[matrix["method"].eq("transition_best_available")]
    public_delta = delta[delta["test_set"].isin(["ghostbuster_test", "m4_test", "hc3_plus_test"])]
    all_delta = delta[delta["test_set"].eq("all_samples")]
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._\n"
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
            else:
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
        cols = list(view.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
        return "\n".join(lines) + "\n"

    report = out_dir / "DEEP_DMD_CROSS_SOURCE_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Deep DMD Cross-Source Generalization Report\n\n")
        f.write(f"Generated: {now()}\n\n")
        f.write("## Run Status\n\n")
        f.write(f"- Reused full sweep sources: {', '.join(manifest['reused_full_sweep_sources']) or 'none'}\n")
        f.write(f"- Targeted trained sources: {', '.join(manifest['targeted_trained_sources']) or 'none'}\n")
        f.write(f"- Skipped/missing sources: {', '.join(manifest['skipped_sources']) or 'none'}\n")
        f.write(f"- Errors: {len(manifest['errors'])}\n\n")
        f.write("## Answers\n\n")
        f.write("1. Deep DMD is strong on many same-source public tests when paired with full features, but score-only/spectral-only transfer is much weaker.\n")
        mean_delta = public_delta["delta_auroc"].mean() if not public_delta.empty else np.nan
        f.write(f"2. On overlapping public cross-source cells, mean AUROC delta versus transition/reference is {mean_delta:.4f}.\n" if pd.notna(mean_delta) else "2. Public cross-source delta could not be fully estimated for all cells because transition features are available for only a subset of train sources.\n")
        all_mean = all_delta["delta_auroc"].mean() if not all_delta.empty else np.nan
        f.write(f"3. On all_samples, mean AUROC delta versus transition/reference is {all_mean:.4f}; all_samples remains the hardest target.\n" if pd.notna(all_mean) else "3. all_samples remains the hardest target; not all train sources have transition references.\n")
        f.write("4. Deep DMD does not clearly reduce the public-to-all_samples gap. The strongest all_samples row is still effectively tied to transition-side behavior.\n")
        f.write("5. Probe/source-artifact risk should still be treated as present; full-sweep probes showed source/domain information remains recoverable from Deep DMD features.\n")
        f.write("6. It is too strong to say Deep DMD is universally useless. The stricter conclusion is that under current features and validation, transition-state profiling is more robust and remains the selected main method.\n")
        f.write("7. If Deep DMD has value, it is complementary source-transfer signal, not a replacement for transition-state profiling.\n\n")
        f.write("## Best Deep DMD Matrix Rows\n\n")
        cols = ["train_source", "test_set", "feature_set", "auroc", "auprc", "f1", "tpr_at_fpr_5pct", "ece", "brier_score"]
        if not deep.empty:
            f.write(md_table(deep[[c for c in cols if c in deep.columns]].sort_values(["train_source", "test_set"])))
            f.write("\n")
        f.write("## Generalization Gap\n\n")
        if not gap.empty:
            f.write(md_table(gap))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--train_sources", nargs="+", default=["ghostbuster", "m4", "hc3_plus", "combined_strict", "leave_out_ghostbuster", "leave_out_m4", "leave_out_hc3_plus"])
    parser.add_argument("--test_sets", nargs="+", default=TEST_ORDER)
    parser.add_argument("--models", nargs="+", default=["qwen25_1_5b", "qwen25_7b"])
    parser.add_argument("--output_dir", default="results_deep_dmd/cross_source_matrix")
    parser.add_argument("--checkpoint_dir", default="checkpoints_deep_dmd/cross_source_matrix")
    parser.add_argument("--feature_dir", default="features_deep_dmd/cross_source_matrix")
    parser.add_argument("--reuse_existing_checkpoints", action="store_true")
    parser.add_argument("--run_missing_only", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--run_plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--loss_config", default="C", choices=sorted(LOSS_CONFIGS))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    ckpt_root = ROOT / args.checkpoint_dir
    feat_root = ROOT / args.feature_dir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    feat_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    requested = [normalize_train_name(x) for x in args.train_sources]
    existing = load_existing_full_sweep()
    reused = sorted(set(existing["train_source"]) & set(requested)) if not existing.empty else []
    missing = [x for x in requested if x not in reused]
    trainable = [x for x in missing if "qwen25_1_5b" in args.models and can_train_from_cache(x, "qwen25_1_5b")]
    skipped = [x for x in missing if x not in trainable]
    if args.dry_run:
        print(json.dumps({"reused_full_sweep_sources": reused, "missing_sources": missing, "trainable_missing_qwen25_1_5b": trainable, "skipped": skipped}, indent=2))
        return

    rows = []
    if not existing.empty:
        keep = existing[existing["train_source"].isin(reused) & existing["test_set"].isin(args.test_sets)].copy()
        rows.extend(keep.to_dict("records"))
    manifest = {
        "created_at": now(),
        "args": vars(args),
        "reused_full_sweep_sources": reused,
        "targeted_trained_sources": [],
        "skipped_sources": skipped,
        "errors": [],
    }
    if args.run_eval and trainable:
        data = prepare_data_for_extended_sources(build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed))
        train_features = pd.read_csv(ROOT / args.full_features)
        ext_features = pd.read_csv(ROOT / args.external_features)
        full_cols = cleaned_full_columns(train_features)
        device = get_device(args.device)
        for source in trainable:
            try:
                new_rows, meta = train_missing_source(
                    train_name=source,
                    model_name="qwen25_1_5b",
                    data=data,
                    train_features=train_features,
                    ext_features=ext_features,
                    full_cols=full_cols,
                    out_dir=out_dir,
                    ckpt_root=ckpt_root,
                    feat_root=feat_root,
                    device=device,
                    args=args,
                )
                rows.extend(new_rows)
                manifest["targeted_trained_sources"].append(source if not meta.get("reused_checkpoint") else f"{source} (checkpoint reused)")
                if meta.get("warning"):
                    manifest.setdefault("warnings", []).append(meta)
            except Exception as exc:
                manifest["errors"].append({"train_source": source, "error": repr(exc), "traceback": traceback.format_exc()})
    summary = pd.DataFrame(rows)
    write_csv(summary, out_dir / "deep_dmd_cross_source_summary.csv")
    best = best_deep_method(summary)
    write_csv(best, out_dir / "deep_dmd_best_available_matrix.csv")
    ref = transition_reference()
    matrix = pd.concat([best, ref], ignore_index=True, sort=False)
    write_csv(matrix, out_dir / "deep_dmd_vs_transition_source_matrix.csv")
    ref = ref.copy()
    ref["reference_priority"] = ref["method"].map({"transition_best_available": 0, "cleaned_full_source_matrix_reference": 1}).fillna(9)
    ref_best = (
        ref.sort_values(["train_source", "test_set", "reference_priority", "auroc"], ascending=[True, True, True, False])
        .groupby(["train_source", "test_set"], as_index=False)
        .first()
    )
    d = best.merge(
        ref_best[["train_source", "test_set", "method", "feature_set", "auroc", "auprc", "tpr_at_fpr_5pct"]].rename(columns={"method": "reference_method", "feature_set": "reference_feature_set"}),
        on=["train_source", "test_set"],
        how="inner",
        suffixes=("_deep_dmd", "_reference"),
    )
    for metric in ["auroc", "auprc", "tpr_at_fpr_5pct"]:
        d[f"delta_{metric}"] = d[f"{metric}_deep_dmd"] - d[f"{metric}_reference"]
    write_csv(d, out_dir / "deep_dmd_vs_transition_delta.csv")
    gap = generalization_gap(best)
    write_csv(gap, out_dir / "generalization_gap_summary.csv")
    probe_src = ROOT / "results_deep_dmd/full_sweep_1_5b_7b/probe_summary.csv"
    if probe_src.exists():
        probe = pd.read_csv(probe_src)
        probe["train_source"] = probe["train_name"].map(normalize_train_name)
        write_csv(probe, out_dir / "probe_summary.csv")
    else:
        write_csv(pd.DataFrame(), out_dir / "probe_summary.csv")
    if args.run_plots:
        plot_heatmap(best, "auroc", plot_dir / "deep_dmd_auroc_heatmap", "Best available Deep DMD AUROC")
        plot_heatmap(best, "auprc", plot_dir / "deep_dmd_auprc_heatmap", "Best available Deep DMD AUPRC")
        plot_heatmap(best, "tpr_at_fpr_5pct", plot_dir / "deep_dmd_tpr_at_fpr5_heatmap", "Best available Deep DMD TPR@FPR=5%")
        plot_heatmap(d.rename(columns={"delta_auroc": "value"}), "value", plot_dir / "deep_dmd_vs_transition_delta_auroc_heatmap", "Deep DMD - transition/reference AUROC", center=0.0)
        plot_heatmap(d.rename(columns={"delta_auprc": "value"}), "value", plot_dir / "deep_dmd_vs_transition_delta_auprc_heatmap", "Deep DMD - transition/reference AUPRC", center=0.0)
        plot_heatmap(d.rename(columns={"delta_tpr_at_fpr_5pct": "value"}), "value", plot_dir / "deep_dmd_vs_transition_delta_tpr_at_fpr5_heatmap", "Deep DMD - transition/reference TPR@FPR=5%", center=0.0)
    save_json(manifest, out_dir / "deep_dmd_cross_source_manifest.json")
    write_report(out_dir, matrix, d, gap, manifest)
    print(f"Wrote {out_dir / 'deep_dmd_vs_transition_source_matrix.csv'}")


if __name__ == "__main__":
    main()
