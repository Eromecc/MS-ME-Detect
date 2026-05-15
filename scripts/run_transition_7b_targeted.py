#!/usr/bin/env python3
"""Targeted qwen25_7b transition profiling vs qwen25_1_5b."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_transition_fullscale_optimized import (  # noqa: E402
    build_base_datasets,
    cleaned_full_columns,
    composite_train_dev,
    ensure_cache,
    eval_one,
    fit_bins_for_train,
    fit_select,
    load_cache_items,
    merge_full_features,
    pca_probe,
    save_fig,
    save_json,
    score_model,
    test_df_for,
    transition_features_for_ids,
)
from run_transition_formal_experiment import model_key_from_name  # noqa: E402
from src import config  # noqa: E402
from src.feature_probability import load_causal_lm, token_loss_cache_path  # noqa: E402
from src.utils import write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rename_transition_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = df.copy()
    ren = {c: f"{c}_{suffix}" for c in out.columns if c != "id"}
    return out.rename(columns=ren)


def load_transition_features(path: Path, suffix: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    return rename_transition_cols(df, suffix) if suffix else df


def make_delta_features(one5: pd.DataFrame, seven: pd.DataFrame) -> pd.DataFrame:
    a = one5.copy()
    b = seven.copy()
    a_cols = set(c for c in a.columns if c != "id")
    delta_cols: dict[str, pd.Series] = {"id": b["id"].reset_index(drop=True)}
    for bc in [c for c in b.columns if c != "id"]:
        ac = bc.replace("qwen25_7b", "qwen25_1_5b")
        if ac in a_cols:
            merged = b[["id", bc]].merge(a[["id", ac]], on="id", how="left")
            delta_cols[f"scale_transition_delta_{bc.replace('qwen25_7b_', '')}"] = (
                pd.to_numeric(merged[bc], errors="coerce") - pd.to_numeric(merged[ac], errors="coerce")
            ).reset_index(drop=True)
    return pd.DataFrame(delta_cols)


def plot_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    all_samples = summary[summary["test_set"].eq("all_samples")].copy()
    if all_samples.empty:
        return
    for metrics, stem, title in [
        (["auroc", "auprc", "f1"], "all_samples_performance", "qwen25_7b targeted transition comparison on all_samples"),
        (["tpr_at_fpr_1pct", "tpr_at_fpr_5pct"], "low_fpr_metrics", "Low-FPR metrics on all_samples"),
    ]:
        long = all_samples[["train_name", "experiment", *metrics]].melt(id_vars=["train_name", "experiment"], var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=long, x="train_name", y="value", hue="experiment", ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8, ncol=2)
        save_fig(fig, plots / stem)
    comp = all_samples[all_samples["experiment"].isin(["full_plus_1_5b_transition", "full_plus_7b_transition", "full_plus_1_5b_and_7b_transition"])]
    if not comp.empty:
        long = comp[["train_name", "experiment", "auroc", "auprc", "f1"]].melt(id_vars=["train_name", "experiment"], var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=long, x="train_name", y="value", hue="experiment", ax=ax)
        ax.set_title("1.5B vs 7B transition comparison")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
        save_fig(fig, plots / "one5b_vs_7b_comparison")


def pca_probe_7b(train_parts: list[pd.DataFrame], trans_cols: list[str], plot_dir: Path) -> dict:
    if not train_parts:
        return {}
    df = pd.concat(train_parts, ignore_index=True).drop_duplicates("id")
    return pca_probe(df, trans_cols, plot_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="qwen25_7b", choices=["qwen25_7b"])
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--one5_transition_root", default="features_transition/fullscale_1_5b_optimized/qwen25_1_5b")
    parser.add_argument("--train_sources", nargs="+", default=["m4", "combined_strict", "leave_out_ghostbuster"])
    parser.add_argument("--test_sets", nargs="+", default=["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"])
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "results_transition" / "qwen25_7b_targeted"
    plot_dir = out_dir / "plots"
    feat_dir = ROOT / "features_transition" / "formal" / args.model_name
    token_dir = ROOT / "features_token_loss"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    data = build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    base_names = ["m4_train", "m4_dev", "m4_test", "ghostbuster_train", "ghostbuster_dev", "ghostbuster_test", "hc3_plus_train", "hc3_plus_dev", "hc3_plus_test", "all_samples"]
    tokenizer = model = None
    if not args.skip_cache:
        local = config.get_model_local_path(model_key_from_name(args.model_name))
        if not config.is_local_model_ready(local):
            raise FileNotFoundError(f"Local 7B model not ready at {local}; not downloading.")
        tokenizer, model = load_causal_lm(local, dtype=config.DTYPE, device_map=args.device_map, local_files_only=True)
    cache_paths = {}
    for name in base_names:
        path = token_loss_cache_path(token_dir, args.model_name, name)
        if args.skip_cache:
            if not path.exists():
                raise FileNotFoundError(path)
        else:
            path = ensure_cache(name, data[name], args.model_name, tokenizer, model, token_dir, args.max_length, args.resume)
        cache_paths[name] = path
    cache_by_dataset = {k: load_cache_items(v) for k, v in cache_paths.items()}

    train_features = pd.read_csv(ROOT / args.full_features)
    ext_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)
    rows = []
    pca_parts = []
    for train_name in args.train_sources:
        train_df, dev_df = composite_train_dev(data, train_name)
        bins = fit_bins_for_train(train_df["id"], cache_by_dataset, feat_dir / train_name / "qwen25_7b_loss_state_bins.json")
        frames7 = {}
        frames15 = {}
        frames_delta = {}
        needed = {**data, f"{train_name}_train": train_df, f"{train_name}_dev": dev_df}
        for name, df in needed.items():
            out_csv = feat_dir / train_name / f"{name}_transition_features.csv"
            regenerate = True
            if out_csv.exists():
                try:
                    regenerate = len(pd.read_csv(out_csv, usecols=["id"])) != len(df)
                except Exception:
                    regenerate = True
            if regenerate:
                transition_features_for_ids(df, cache_by_dataset, bins, args.model_name, out_csv)
            frames7[name] = pd.read_csv(out_csv)
            one5_path = ROOT / args.one5_transition_root / train_name / f"{name}_transition_features.csv"
            if not one5_path.exists():
                # Composite 1.5B files use the same names for train/dev composites.
                one5_path = ROOT / args.one5_transition_root / train_name / f"{name.replace(train_name + '_', '')}_transition_features.csv"
            if one5_path.exists():
                frames15[name] = pd.read_csv(one5_path)
                frames_delta[name] = make_delta_features(frames15[name], frames7[name])
        train_base = merge_full_features(train_df, train_features, ext_features, full_cols)
        dev_base = merge_full_features(dev_df, train_features, ext_features, full_cols)
        train_key = f"{train_name}_train"
        dev_key = f"{train_name}_dev"
        for frame_map, suffix in [(frames15, "one5"), (frames7, "seven"), (frames_delta, "delta")]:
            if train_key in frame_map:
                train_base = train_base.merge(rename_transition_cols(frame_map[train_key], suffix), on="id", how="left")
            if dev_key in frame_map:
                dev_base = dev_base.merge(rename_transition_cols(frame_map[dev_key], suffix), on="id", how="left")
        trans15 = [c for c in train_base.columns if c.endswith("_one5")]
        trans7 = [c for c in train_base.columns if c.endswith("_seven")]
        delta_cols = [c for c in train_base.columns if c.endswith("_delta") or c.startswith("scale_transition_delta_")]
        pca_parts.append(train_base[["id", "label", "source_dataset", "domain", "transition_split", *trans7]].copy())
        experiments = {
            "full_without_transition": full_cols,
            "full_plus_1_5b_transition": full_cols + trans15,
            "full_plus_7b_transition": full_cols + trans7,
            "full_plus_1_5b_and_7b_transition": full_cols + trans15 + trans7,
        }
        if delta_cols:
            experiments["transition_scale_response_1_5b_to_7b"] = delta_cols
            experiments["full_plus_transition_scale_response_1_5b_to_7b"] = full_cols + delta_cols
        for exp, cols in experiments.items():
            # Drop missing columns if 1.5B comparison files are absent.
            cols = [c for c in cols if c in train_base.columns]
            if not cols:
                continue
            best_name, clf, med, _ = fit_select(train_base, dev_base, cols)
            for test_name in args.test_sets:
                test_meta = data[test_name]
                test_base = merge_full_features(test_meta, train_features, ext_features, full_cols)
                for frame_map, suffix in [(frames15, "one5"), (frames7, "seven"), (frames_delta, "delta")]:
                    if test_name in frame_map:
                        test_base = test_base.merge(rename_transition_cols(frame_map[test_name], suffix), on="id", how="left")
                metrics = eval_one(clf, test_base, cols, med, out_dir / f"{train_name}_{exp}_to_{test_name}")
                rows.append({
                    "train_name": train_name,
                    "experiment": exp,
                    "test_set": test_name,
                    "best_model": best_name,
                    "n_train": len(train_base),
                    "n_test": len(test_base),
                    "n_features": len(cols),
                    "auroc": metrics.get("auroc", np.nan),
                    "auprc": metrics.get("auprc", np.nan),
                    "f1": metrics.get("f1", np.nan),
                    "tpr_at_fpr_1pct": metrics.get("tpr_at_fpr_1pct", np.nan),
                    "tpr_at_fpr_5pct": metrics.get("tpr_at_fpr_5pct", np.nan),
                    "ece": metrics.get("expected_calibration_error", metrics.get("ECE", np.nan)),
                    "brier_score": metrics.get("brier_score", np.nan),
                })
    summary = pd.DataFrame(rows)
    write_csv(summary, out_dir / "transition_7b_summary.csv")
    comparison = summary[summary["experiment"].isin(["full_without_transition", "full_plus_1_5b_transition", "full_plus_7b_transition", "full_plus_1_5b_and_7b_transition"])]
    write_csv(comparison, out_dir / "transition_7b_vs_1_5b_comparison.csv")
    plot_bars(summary, out_dir)
    probes = pca_probe_7b(pca_parts, [c for c in pca_parts[0].columns if c.endswith("_seven")] if pca_parts else [], plot_dir)
    cache_info = {}
    for name, path in cache_paths.items():
        man = path.with_name(path.name.replace(".jsonl.gz", "_manifest.json"))
        cache_info[name] = json.loads(man.read_text()) if man.exists() else {"cache_path": str(path)}
    save_json({
        "created_at": now(),
        "args": vars(args),
        "cache_info": cache_info,
        "pca_probe": probes,
        "note": "Only qwen25_7b was run; qwen25_14b was not used.",
    }, out_dir / "transition_7b_manifest.json")
    print(f"Wrote {out_dir / 'transition_7b_summary.csv'}")


if __name__ == "__main__":
    main()
