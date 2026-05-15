#!/usr/bin/env python3
"""Formal transition-state profiling experiment with cached token losses."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
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
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.feature_probability import load_causal_lm, read_cached_ids, text_hash, token_loss_cache_path, token_loss_sequence, write_token_loss_manifest
from src.feature_transition_profile import (
    build_transition_features_from_loss_cache,
    fit_loss_bins,
    load_loss_sequences,
)
from src.train_eval import detector_metrics, probabilities, save_calibration_curve, save_pr_curve, save_roc_curve
from src.utils import write_csv

sns.set_theme(style="whitegrid", context="talk")
META = {"id", "text", "label", "source_dataset", "language", "domain", "generator", "attack_type", "split", "type", "source", "topic", "transition_split"}


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


def model_key_from_name(model_name: str) -> str:
    mapping = {"qwen25_1_5b": "small", "qwen25_7b": "medium", "qwen25_14b": "large"}
    if model_name not in mapping:
        raise ValueError(f"unsupported model_name={model_name}; supported: {sorted(mapping)}")
    return mapping[model_name]


def load_source_split(source_splits: Path, source: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(source_splits / f"{source}_strict_{split}.csv")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return df


def sample_df(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df.copy()
    if "label" in df.columns and df["label"].nunique() > 1:
        return df.groupby("label", group_keys=False).apply(lambda g: g.sample(n=max(1, int(round(max_rows * len(g) / len(df)))), random_state=seed)).head(max_rows).copy()
    return df.sample(n=max_rows, random_state=seed).copy()


def build_datasets(source_splits: Path, external_test: Path, max_rows: int | None, seed: int) -> dict[str, pd.DataFrame]:
    data = {}
    for source in ["ghostbuster", "m4", "hc3_plus"]:
        for split in ["train", "dev", "test"]:
            key = f"{source}_{split}"
            data[key] = sample_df(load_source_split(source_splits, source, split), max_rows, seed)
            data[key]["transition_split"] = key
    data["combined_strict_train"] = pd.concat([data["ghostbuster_train"], data["m4_train"], data["hc3_plus_train"]], ignore_index=True)
    data["combined_strict_dev"] = pd.concat([data["ghostbuster_dev"], data["m4_dev"], data["hc3_plus_dev"]], ignore_index=True)
    data["combined_strict_test"] = pd.concat([data["ghostbuster_test"], data["m4_test"], data["hc3_plus_test"]], ignore_index=True)
    data["leave_out_ghostbuster_train"] = pd.concat([data["m4_train"], data["hc3_plus_train"]], ignore_index=True)
    data["leave_out_ghostbuster_dev"] = pd.concat([data["m4_dev"], data["hc3_plus_dev"]], ignore_index=True)
    data["all_samples"] = sample_df(pd.read_csv(external_test), max_rows, seed)
    data["all_samples"]["label"] = pd.to_numeric(data["all_samples"]["label"], errors="coerce").astype(int)
    data["all_samples"]["transition_split"] = "all_samples"
    return data


def dataset_names_needed(train_sources: list[str], test_sets: list[str]) -> list[str]:
    names = set()
    for train in train_sources:
        if train == "m4":
            names.update(["m4_train", "m4_dev"])
        elif train == "combined_strict":
            names.update(["combined_strict_train", "combined_strict_dev"])
        elif train == "leave_out_ghostbuster":
            names.update(["leave_out_ghostbuster_train", "leave_out_ghostbuster_dev"])
    for test in test_sets:
        names.add(test.replace("_strict", "").replace("_test", "_test") if test != "all_samples" else "all_samples")
    return sorted(names)


def cache_has_ids(cache_path: Path, ids: set[str]) -> bool:
    return cache_path.exists() and ids.issubset(read_cached_ids(cache_path))


def ensure_token_loss_cache(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    model_name: str,
    tokenizer,
    model,
    output_dir: Path,
    max_length: int,
    resume: bool,
) -> Path:
    cache_path = token_loss_cache_path(output_dir, model_name, dataset_name)
    manifest_path = cache_path.with_name(cache_path.name.replace(".jsonl.gz", "_manifest.json"))
    ids = set(df["id"].astype(str))
    if resume and cache_has_ids(cache_path, ids):
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = read_cached_ids(cache_path) if resume else set()
    token_counts: list[int] = []
    skipped_ids: list[str] = []
    failed_ids: list[str] = []
    with gzip.open(cache_path, "at" if resume else "wt", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            rid = str(row["id"])
            if rid in cached:
                skipped_ids.append(rid)
                continue
            try:
                losses = token_loss_sequence(row["text"], tokenizer, model, max_length=max_length)
                item = {
                    "id": row["id"],
                    "model_name": model_name,
                    "token_count": int(losses.size),
                    "loss_sequence": [float(x) for x in losses.tolist()],
                    "rank_sequence": pd.Series(-losses).rank(method="average", pct=True).round(8).tolist() if losses.size else [],
                    "prob_sequence": [float(math.exp(-min(float(x), 50.0))) for x in losses.tolist()],
                    "text_hash": text_hash(row["text"]),
                    "max_length": int(max_length),
                    "created_at": now(),
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                token_counts.append(int(losses.size))
            except Exception:
                failed_ids.append(rid)
    write_token_loss_manifest(
        manifest_path,
        model_name=model_name,
        dataset_name=dataset_name,
        cache_path=cache_path,
        token_counts=token_counts,
        max_length=max_length,
        skipped_ids=skipped_ids,
        failed_ids=failed_ids,
    )
    return cache_path


def cleaned_full_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if "32b" not in c.lower()]
    cols = [c for c in cols if c.startswith(("burst_", "struct_", "scale_", "qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_"))]
    work = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    constant = work.nunique(dropna=True) <= 1
    all_nan = work.isna().all()
    return [c for c in cols if not constant.get(c, False) and not all_nan.get(c, False)]


def make_xy(df: pd.DataFrame, cols: list[str], medians: dict | None = None):
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(medians).fillna(0.0)
    y = pd.to_numeric(df["label"], errors="coerce").astype(int)
    return x, y, {k: float(v) for k, v in medians.items()}


def candidates():
    out = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced", n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=config.RANDOM_STATE)
    except Exception:
        pass
    return out


def fit_select(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str]):
    x_train, y_train, med = make_xy(train_df, cols)
    x_dev, y_dev, _ = make_xy(dev_df, cols, med)
    rows = []
    best = None
    for name, model in candidates().items():
        model.fit(x_train, y_train)
        prob = probabilities(model, x_dev)
        pred = (prob >= 0.5).astype(int)
        m = {"model": name}
        m.update(detector_metrics(y_dev, prob, y_pred=pred))
        rows.append(m)
        rank = (m.get("auprc", -np.inf), m.get("auroc", -np.inf), m.get("f1", -np.inf))
        if best is None or rank > best[0]:
            best = (rank, name, model, med)
    return best[1], best[2], best[3], pd.DataFrame(rows)


def eval_one(model, df: pd.DataFrame, cols: list[str], med: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, _ = make_xy(df, cols, med)
    prob = probabilities(model, x)
    pred = (prob >= 0.5).astype(int)
    m = detector_metrics(y, prob, y_pred=pred)
    write_csv(pd.DataFrame([m]), out_dir / "metrics.csv")
    write_csv(save_roc_curve(y, prob, out_dir), out_dir / "roc_curve.csv")
    write_csv(save_pr_curve(y, prob, out_dir), out_dir / "pr_curve.csv")
    write_csv(save_calibration_curve(y, prob, out_dir), out_dir / "calibration_bins.csv")
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(classification_report(y, pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
    keep = [c for c in ["id", "label", "source_dataset", "domain", "generator", "transition_split"] if c in df.columns]
    preds = df[keep].copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    write_csv(preds, out_dir / "predictions.csv")
    return m


def merge_full_features(meta: pd.DataFrame, train_features: pd.DataFrame, external_features: pd.DataFrame, full_cols: list[str]) -> pd.DataFrame:
    ids = set(meta["id"].astype(str))
    train = train_features[["id", *full_cols]]
    ext = external_features[["id", *full_cols]]
    combined = pd.concat([train, ext], ignore_index=True).drop_duplicates("id")
    out = meta.merge(combined, on="id", how="left")
    return out


def plot_pca(all_df: pd.DataFrame, trans_cols: list[str], plot_dir: Path) -> dict:
    if len(all_df) < 5:
        return {}
    x, y, _ = make_xy(all_df, trans_cols)
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(x))
    p = all_df.copy()
    p["PC1"] = coords[:, 0]
    p["PC2"] = coords[:, 1]
    p["label_name"] = y.map({0: "Human", 1: "AI"})
    for hue, stem, title in [
        ("label_name", "formal_transition_pca_by_label", "Transition PCA by label"),
        ("source_dataset", "formal_transition_pca_by_source", "Transition PCA by source"),
        ("domain", "formal_transition_pca_by_domain", "Transition PCA by domain"),
    ]:
        if hue not in p.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=p, x="PC1", y="PC2", hue=hue, alpha=0.75, s=35, ax=ax)
        ax.set_title(title)
        save_fig(fig, plot_dir / stem)
    return {}


def plot_summaries(summary: pd.DataFrame, plot_dir: Path) -> None:
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        return
    all_samples = ok[ok["test_set"].eq("all_samples")].copy()
    if not all_samples.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=all_samples, x="train_name", y="auroc", hue="experiment", ax=ax)
        ax.set_title("Formal transition ablation on all_samples")
        ax.tick_params(axis="x", rotation=25)
        save_fig(fig, plot_dir / "formal_transition_ablation_barplot")
    trans_only = ok[ok["experiment"].eq("transition_only")]
    if not trans_only.empty:
        pivot = trans_only.pivot_table(index="train_name", columns="test_set", values="auroc")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Transition-only AUROC source matrix")
        save_fig(fig, plot_dir / "transition_only_source_matrix_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="qwen25_1_5b", choices=["qwen25_1_5b", "qwen25_7b", "qwen25_14b"])
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--train_sources", nargs="+", default=["m4", "combined_strict", "leave_out_ghostbuster"])
    parser.add_argument("--test_sets", nargs="+", default=["m4_test", "ghostbuster_test", "hc3_plus_test", "all_samples"])
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--save_token_loss", action="store_true")
    parser.add_argument("--run_transition", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result_dir = ROOT / "results_transition"
    plot_dir = result_dir / "plots"
    token_dir = ROOT / "features_token_loss"
    trans_root = ROOT / "features_transition" / "formal" / args.model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    trans_root.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    needed = dataset_names_needed(args.train_sources, args.test_sets)
    dry_info = {"needed_datasets": {k: len(datasets[k]) for k in needed if k in datasets}, "model_name": args.model_name, "max_rows_per_split": args.max_rows_per_split}
    if args.dry_run:
        save_json(dry_info, result_dir / "formal_transition_dry_run.json")
        print(json.dumps(dry_info, indent=2))
        return

    model_key = model_key_from_name(args.model_name)
    tokenizer = model = None
    if args.save_token_loss:
        local = config.get_model_local_path(model_key)
        if not config.is_local_model_ready(local):
            raise FileNotFoundError(f"Local model not ready at {local}; not downloading.")
        tokenizer, model = load_causal_lm(local, dtype=config.DTYPE, device_map=None, local_files_only=True)
    cache_paths: dict[str, Path] = {}
    cache_manifests = {}
    for name in needed:
        if name not in datasets:
            continue
        cache_path = token_loss_cache_path(token_dir, args.model_name, name)
        if args.save_token_loss:
            cache_path = ensure_token_loss_cache(name, datasets[name], model_name=args.model_name, tokenizer=tokenizer, model=model, output_dir=token_dir, max_length=args.max_length, resume=args.resume)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing token loss cache for {name}: {cache_path}")
        cache_paths[name] = cache_path
        man_path = cache_path.with_name(cache_path.name.replace(".jsonl.gz", "_manifest.json"))
        cache_manifests[name] = json.loads(man_path.read_text()) if man_path.exists() else {}

    rows = []
    all_pca_parts = []
    train_features = pd.read_csv(ROOT / args.full_features)
    external_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)
    for train_name in args.train_sources:
        train_key = f"{train_name}_train" if train_name in ["m4"] else f"{train_name}_train"
        dev_key = f"{train_name}_dev" if train_name in ["m4"] else f"{train_name}_dev"
        train_df = datasets[train_key]
        dev_df = datasets[dev_key]
        train_losses = load_loss_sequences(cache_paths[train_key])
        bins = fit_loss_bins([train_losses[str(x)] for x in train_df["id"].astype(str) if str(x) in train_losses], [3, 5, 7])
        bin_path = trans_root / train_name / f"{args.model_name}_loss_state_bins.json"
        save_json({"model_name": args.model_name, "train_name": train_name, "bins_by_state": bins, "created_at": now()}, bin_path)
        feature_frames = {}
        for ds_name in set([train_key, dev_key] + [t if t == "all_samples" else t for t in args.test_sets]):
            if ds_name not in cache_paths:
                continue
            out_csv = trans_root / train_name / f"{ds_name}_{args.model_name}_transition_features.csv"
            if args.run_transition or not out_csv.exists():
                build_transition_features_from_loss_cache(cache_paths[ds_name], out_csv, model_name=args.model_name, bins_by_state=bins)
            feature_frames[ds_name] = pd.read_csv(out_csv)
        trans_cols = [c for c in feature_frames[train_key].columns if c != "id" and pd.api.types.is_numeric_dtype(feature_frames[train_key][c])]
        train_base = merge_full_features(train_df, train_features, external_features, full_cols).merge(feature_frames[train_key], on="id", how="left")
        dev_base = merge_full_features(dev_df, train_features, external_features, full_cols).merge(feature_frames[dev_key], on="id", how="left")
        all_pca_parts.append(train_base[["id", "label", "source_dataset", "domain", *trans_cols]].copy())
        experiments = {
            "transition_only": trans_cols,
            "full_cleaned_without_transition": full_cols,
            "full_cleaned_plus_transition": full_cols + trans_cols,
        }
        if args.run_eval:
            for exp, cols in experiments.items():
                best_name, clf, med, dev_metrics = fit_select(train_base, dev_base, cols)
                joblib.dump(clf, result_dir / f"formal_{train_name}_{exp}_best_model.joblib")
                for test_set in args.test_sets:
                    ds_key = test_set
                    test_df = datasets[ds_key]
                    feat_key = ds_key
                    test_base = merge_full_features(test_df, train_features, external_features, full_cols).merge(feature_frames[feat_key], on="id", how="left")
                    metrics = eval_one(clf, test_base, cols, med, result_dir / f"formal_{train_name}_{exp}_to_{test_set}")
                    rows.append({
                        "train_name": train_name,
                        "experiment": exp,
                        "test_set": test_set,
                        "status": "ok",
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
    if not summary.empty:
        write_csv(summary, result_dir / "formal_transition_summary.csv")
        write_csv(summary, result_dir / "formal_transition_source_matrix.csv")
        plot_summaries(summary, plot_dir)
    if all_pca_parts:
        pca_df = pd.concat(all_pca_parts, ignore_index=True).drop_duplicates("id")
        pca_cols = [c for c in pca_df.columns if c not in META and pd.api.types.is_numeric_dtype(pca_df[c])]
        plot_pca(pca_df, pca_cols, plot_dir)
    save_json({
        "created_at": now(),
        "args": vars(args),
        "dry_info": dry_info,
        "cache_manifests": cache_manifests,
        "full_cleaned_feature_count": len(full_cols),
    }, result_dir / "formal_transition_manifest.json")
    print(f"Wrote {result_dir / 'formal_transition_summary.csv'}")


if __name__ == "__main__":
    main()
