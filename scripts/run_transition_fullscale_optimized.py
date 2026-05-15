#!/usr/bin/env python3
"""Full-scale qwen25_1_5b transition optimization and late fusion."""

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
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_transition_formal_experiment import (  # noqa: E402
    META,
    cleaned_full_columns,
    eval_one,
    fit_select,
    make_xy,
    merge_full_features,
    model_key_from_name,
    save_fig,
)
from src import config  # noqa: E402
from src.feature_probability import (  # noqa: E402
    load_causal_lm,
    read_cached_ids,
    text_hash,
    token_loss_cache_path,
    token_loss_sequence,
    write_token_loss_manifest,
)
from src.feature_transition_profile import (  # noqa: E402
    fit_loss_bins,
    load_loss_sequences,
    transition_features_from_states,
)
from src.train_eval import detector_metrics, probabilities  # noqa: E402
from src.utils import write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_source_split(source_splits: Path, source: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(source_splits / f"{source}_strict_{split}.csv")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    df["transition_split"] = f"{source}_{split}"
    return df


def sample_df(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df.copy()
    if "label" in df.columns and df["label"].nunique() > 1:
        out = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=max(1, int(round(max_rows * len(g) / len(df)))), random_state=seed)
        )
        return out.head(max_rows).copy()
    return df.sample(n=max_rows, random_state=seed).copy()


def build_base_datasets(source_splits: Path, external_test: Path, max_rows: int | None, seed: int) -> dict[str, pd.DataFrame]:
    data = {}
    for source in ["ghostbuster", "m4", "hc3_plus"]:
        for split in ["train", "dev", "test"]:
            key = f"{source}_{split}"
            data[key] = sample_df(load_source_split(source_splits, source, split), max_rows, seed)
    data["all_samples"] = sample_df(pd.read_csv(external_test), max_rows, seed)
    data["all_samples"]["label"] = pd.to_numeric(data["all_samples"]["label"], errors="coerce").astype(int)
    data["all_samples"]["transition_split"] = "all_samples"
    return data


def composite_train_dev(data: dict[str, pd.DataFrame], train_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_name == "m4":
        return data["m4_train"], data["m4_dev"]
    if train_name == "leave_out_ghostbuster":
        return pd.concat([data["m4_train"], data["hc3_plus_train"]], ignore_index=True), pd.concat([data["m4_dev"], data["hc3_plus_dev"]], ignore_index=True)
    if train_name == "combined_strict":
        return (
            pd.concat([data["ghostbuster_train"], data["m4_train"], data["hc3_plus_train"]], ignore_index=True),
            pd.concat([data["ghostbuster_dev"], data["m4_dev"], data["hc3_plus_dev"]], ignore_index=True),
        )
    raise ValueError(f"unknown train_name={train_name}")


def test_df_for(data: dict[str, pd.DataFrame], test_name: str) -> pd.DataFrame:
    mapping = {
        "m4_test": "m4_test",
        "ghostbuster_test": "ghostbuster_test",
        "hc3_plus_test": "hc3_plus_test",
        "all_samples": "all_samples",
    }
    return data[mapping[test_name]]


def cache_has_ids(cache_path: Path, ids: set[str]) -> bool:
    return cache_path.exists() and ids.issubset(read_cached_ids(cache_path))


def ensure_cache(dataset_name: str, df: pd.DataFrame, model_name: str, tokenizer, model, out_dir: Path, max_length: int, resume: bool) -> Path:
    cache_path = token_loss_cache_path(out_dir, model_name, dataset_name)
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
        cache_path.with_name(cache_path.name.replace(".jsonl.gz", "_manifest.json")),
        model_name=model_name,
        dataset_name=dataset_name,
        cache_path=cache_path,
        token_counts=token_counts,
        max_length=max_length,
        skipped_ids=skipped_ids,
        failed_ids=failed_ids,
    )
    return cache_path


def load_cache_items(cache_path: Path) -> dict[str, np.ndarray]:
    return load_loss_sequences(cache_path)


def states_from_bins(vals: np.ndarray, bins: list[float], n_states: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=int)
    if not bins:
        return np.zeros(vals.size, dtype=int)
    return np.clip(np.searchsorted(np.asarray(bins), vals, side="right"), 0, n_states - 1).astype(int)


def zscore(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    sd = vals.std()
    return (vals - vals.mean()) / (sd if sd > 0 else 1.0)


def rank_states(vals: np.ndarray, n_states: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=int)
    ranks = pd.Series(vals).rank(method="average", pct=True).to_numpy()
    return np.clip(np.floor(ranks * n_states).astype(int), 0, n_states - 1)


def fit_bins_for_train(train_ids: pd.Series, cache_by_dataset: dict[str, dict[str, np.ndarray]], bins_out: Path) -> dict:
    seqs = []
    zseqs = []
    for rid in train_ids.astype(str):
        for cache in cache_by_dataset.values():
            if rid in cache:
                vals = cache[rid]
                seqs.append(vals)
                zseqs.append(zscore(vals))
                break
    raw_bins = fit_loss_bins(seqs, [3, 5, 7])
    z_bins = fit_loss_bins(zseqs, [3, 5, 7])
    bins = {"raw": raw_bins, "zscore": z_bins}
    save_json({"created_at": now(), "bins": bins}, bins_out)
    return bins


def transition_features_for_ids(df: pd.DataFrame, cache_by_dataset: dict[str, dict[str, np.ndarray]], bins: dict, model_name: str, output_path: Path) -> pd.DataFrame:
    rows = []
    for rid in df["id"].astype(str):
        losses = None
        for cache in cache_by_dataset.values():
            if rid in cache:
                losses = cache[rid]
                break
        feats = {"id": rid}
        if losses is None:
            rows.append(feats)
            continue
        for n in [3, 5, 7]:
            feats.update(transition_features_from_states(states_from_bins(losses, bins["raw"][str(n)], n), f"{model_name}_raw_lossq{n}", n))
            feats.update(transition_features_from_states(states_from_bins(zscore(losses), bins["zscore"][str(n)], n), f"{model_name}_z_lossq{n}", n))
            # Low rank = low loss; high rank = high loss.
            feats.update(transition_features_from_states(rank_states(losses, n), f"{model_name}_doc_rank{n}", n))
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def select_transition_features(train_df: pd.DataFrame, dev_df: pd.DataFrame, trans_cols: list[str], mode: str, seed: int) -> list[str]:
    if mode == "transition_all":
        return trans_cols
    if mode == "transition_summary_only":
        return [c for c in trans_cols if "_trans_" not in c]
    x_train, y_train, med = make_xy(train_df, trans_cols)
    if mode == "transition_l1_selected":
        clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", penalty="l1", solver="liblinear", C=0.1))])
        clf.fit(x_train, y_train)
        coef = np.abs(clf.named_steps["model"].coef_[0])
        selected = [c for c, v in zip(trans_cols, coef) if v > 1e-9]
        return selected or trans_cols[: min(50, len(trans_cols))]
    n = int(mode.replace("transition_top", ""))
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(x_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1][: min(n, len(trans_cols))]
    return [trans_cols[i] for i in order]


def fit_model_scores(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str]):
    name, model, med, metrics = fit_select(train_df, dev_df, cols)
    _, y_dev, _ = make_xy(dev_df, cols, med)
    prob_dev = probabilities(model, make_xy(dev_df, cols, med)[0])
    return name, model, med, prob_dev, y_dev


def score_model(model, df: pd.DataFrame, cols: list[str], med: dict) -> tuple[np.ndarray, np.ndarray]:
    x, y, _ = make_xy(df, cols, med)
    return probabilities(model, x), y.to_numpy(dtype=int)


def eval_scores(y_true, score) -> dict:
    pred = (np.asarray(score) >= 0.5).astype(int)
    return detector_metrics(y_true, score, y_pred=pred)


def eval_scores(y_true, score) -> dict:
    pred = (np.asarray(score) >= 0.5).astype(int)
    return detector_metrics(y_true, score, y_pred=pred)


def choose_alpha(y, full_score, trans_score) -> tuple[float, list[dict]]:
    rows = []
    best = None
    for alpha in np.linspace(0, 1, 11):
        score = alpha * full_score + (1 - alpha) * trans_score
        m = eval_scores(y, score)
        row = {"alpha": float(alpha), **m}
        rows.append(row)
        rank = (m.get("auprc", -np.inf), m.get("auroc", -np.inf), m.get("f1", -np.inf))
        if best is None or rank > best[0]:
            best = (rank, float(alpha))
    return best[1], rows


def score_model(model, df: pd.DataFrame, cols: list[str], med: dict) -> tuple[np.ndarray, np.ndarray]:
    x, y, _ = make_xy(df, cols, med)
    return probabilities(model, x), y.to_numpy(dtype=int)


def plot_outputs(summary: pd.DataFrame, fusion: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    all_samples = summary[summary["test_set"].eq("all_samples")]
    if not all_samples.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=all_samples, x="train_name", y="auroc", hue="experiment", ax=ax)
        ax.set_title("Full-scale 1.5B transition variants on all_samples")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, plot_dir / "fullscale_transition_all_samples_auroc")
    if not fusion.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=fusion[fusion["test_set"].eq("all_samples")], x="train_name", y="auroc", hue="method", ax=ax)
        ax.set_title("Late fusion vs direct concat on all_samples")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, plot_dir / "late_fusion_all_samples")


def pca_probe(df: pd.DataFrame, trans_cols: list[str], plot_dir: Path) -> dict:
    if len(df) < 10:
        return {}
    sample = df.groupby("transition_split", group_keys=False).apply(lambda g: g.sample(n=min(1000, len(g)), random_state=42)).reset_index(drop=True)
    x, y, _ = make_xy(sample, trans_cols)
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(x))
    p = sample.copy()
    p["PC1"] = coords[:, 0]
    p["PC2"] = coords[:, 1]
    p["label_name"] = y.map({0: "Human", 1: "AI"})
    for hue, name in [("label_name", "pca_by_label"), ("source_dataset", "pca_by_source"), ("domain", "pca_by_domain")]:
        if hue not in p.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=p, x="PC1", y="PC2", hue=hue, alpha=0.65, s=24, ax=ax)
        ax.set_title(f"Transition PCA by {hue}")
        save_fig(fig, plot_dir / name)
    probes = {}
    for target, col in [("label", "label"), ("source", "source_dataset"), ("domain", "domain")]:
        if col not in sample.columns or sample[col].nunique() < 2:
            continue
        try:
            x_all, _, _ = make_xy(sample, trans_cols)
            y_target = sample[col].astype(str) if col != "label" else sample[col].astype(int)
            clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            clf.fit(x_all, y_target)
            probes[f"{target}_in_sample_probe_accuracy"] = float(clf.score(x_all, y_target))
        except Exception as exc:
            probes[f"{target}_probe_error"] = str(exc)
    return probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--model_name", default="qwen25_1_5b", choices=["qwen25_1_5b"])
    parser.add_argument("--train_sources", nargs="+", default=["m4", "leave_out_ghostbuster", "combined_strict"])
    parser.add_argument("--test_sets", nargs="+", default=["m4_test", "ghostbuster_test", "hc3_plus_test", "all_samples"])
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "results_transition" / "fullscale_1_5b_optimized"
    plot_dir = out_dir / "plots"
    feat_dir = ROOT / "features_transition" / "fullscale_1_5b_optimized" / args.model_name
    token_dir = ROOT / "features_token_loss"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    data = build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    needed_base = ["ghostbuster_train", "ghostbuster_dev", "ghostbuster_test", "m4_train", "m4_dev", "m4_test", "hc3_plus_train", "hc3_plus_dev", "hc3_plus_test", "all_samples"]
    tokenizer = model = None
    if not args.skip_cache:
        local = config.get_model_local_path(model_key_from_name(args.model_name))
        if not config.is_local_model_ready(local):
            raise FileNotFoundError(f"Local 1.5B model not ready at {local}; not downloading.")
        tokenizer, model = load_causal_lm(local, dtype=config.DTYPE, device_map=None, local_files_only=True)
    cache_paths = {}
    for name in needed_base:
        df = data[name]
        path = token_loss_cache_path(token_dir, args.model_name, name)
        if args.skip_cache:
            if not path.exists():
                raise FileNotFoundError(path)
        else:
            path = ensure_cache(name, df, args.model_name, tokenizer, model, token_dir, args.max_length, args.resume)
        cache_paths[name] = path
    cache_by_dataset = {k: load_cache_items(v) for k, v in cache_paths.items()}

    train_features = pd.read_csv(ROOT / args.full_features)
    ext_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)
    transition_modes = ["transition_all", "transition_top20", "transition_top50", "transition_top100", "transition_summary_only", "transition_l1_selected"]
    rows = []
    fs_rows = []
    fusion_rows = []
    pca_parts = []
    for train_name in args.train_sources:
        train_df, dev_df = composite_train_dev(data, train_name)
        bins = fit_bins_for_train(train_df["id"], cache_by_dataset, feat_dir / train_name / "qwen25_1_5b_loss_state_bins.json")
        frames = {}
        for name, df in {**data, f"{train_name}_train": train_df, f"{train_name}_dev": dev_df}.items():
            out_csv = feat_dir / train_name / f"{name}_transition_features.csv"
            if not out_csv.exists():
                transition_features_for_ids(df, cache_by_dataset, bins, args.model_name, out_csv)
            frames[name] = pd.read_csv(out_csv)
        train_base = merge_full_features(train_df, train_features, ext_features, full_cols).merge(frames[f"{train_name}_train"], on="id", how="left")
        dev_base = merge_full_features(dev_df, train_features, ext_features, full_cols).merge(frames[f"{train_name}_dev"], on="id", how="left")
        trans_cols = [c for c in frames[f"{train_name}_train"].columns if c != "id" and pd.api.types.is_numeric_dtype(frames[f"{train_name}_train"][c])]
        pca_parts.append(train_base[["id", "label", "source_dataset", "domain", "transition_split", *trans_cols]])

        selected_by_mode = {mode: select_transition_features(train_base, dev_base, trans_cols, mode, args.seed) for mode in transition_modes}
        for mode, cols in selected_by_mode.items():
            fs_rows.append({"train_name": train_name, "feature_set": mode, "n_features": len(cols)})

        experiment_cols = {
            "transition_only": trans_cols,
            "full_without_transition": full_cols,
            "full_plus_transition": full_cols + trans_cols,
            **selected_by_mode,
        }
        fitted = {}
        for exp, cols in experiment_cols.items():
            best_name, clf, med, _ = fit_select(train_base, dev_base, cols)
            fitted[exp] = (best_name, clf, med, cols)
            for test_name in args.test_sets:
                test_meta = test_df_for(data, test_name)
                test_base = merge_full_features(test_meta, train_features, ext_features, full_cols).merge(frames[test_name], on="id", how="left")
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

        # Late fusion: choose alpha on dev only.
        full_name, full_model, full_med, full_cols_used = fitted["full_without_transition"]
        trans_name, trans_model, trans_med, trans_cols_used = fitted["transition_only"]
        full_dev, y_dev = score_model(full_model, dev_base, full_cols_used, full_med)
        trans_dev, _ = score_model(trans_model, dev_base, trans_cols_used, trans_med)
        alpha, alpha_rows = choose_alpha(y_dev, full_dev, trans_dev)
        for r in alpha_rows:
            r.update({"train_name": train_name, "split": "dev"})
        for test_name in args.test_sets:
            test_meta = test_df_for(data, test_name)
            test_base = merge_full_features(test_meta, train_features, ext_features, full_cols).merge(frames[test_name], on="id", how="left")
            full_score, y_test = score_model(full_model, test_base, full_cols_used, full_med)
            trans_score, _ = score_model(trans_model, test_base, trans_cols_used, trans_med)
            fused = alpha * full_score + (1 - alpha) * trans_score
            for method, score in [("full", full_score), ("transition", trans_score), ("late_fusion", fused)]:
                m = eval_scores(y_test, score)
                fusion_rows.append({
                    "train_name": train_name,
                    "method": method,
                    "test_set": test_name,
                    "alpha": alpha if method == "late_fusion" else np.nan,
                    "auroc": m.get("auroc", np.nan),
                    "auprc": m.get("auprc", np.nan),
                    "f1": m.get("f1", np.nan),
                    "tpr_at_fpr_1pct": m.get("tpr_at_fpr_1pct", np.nan),
                    "tpr_at_fpr_5pct": m.get("tpr_at_fpr_5pct", np.nan),
                    "ece": m.get("expected_calibration_error", m.get("ECE", np.nan)),
                    "brier_score": m.get("brier_score", np.nan),
                })

    summary = pd.DataFrame(rows)
    fusion = pd.DataFrame(fusion_rows)
    fs = pd.DataFrame(fs_rows)
    write_csv(summary, out_dir / "transition_optimized_summary.csv")
    write_csv(fusion, out_dir / "late_fusion_summary.csv")
    write_csv(fs, out_dir / "feature_selection_summary.csv")
    probes = pca_probe(pd.concat(pca_parts, ignore_index=True).drop_duplicates("id"), trans_cols, plot_dir) if pca_parts else {}
    plot_outputs(summary, fusion, plot_dir)
    save_json({
        "created_at": now(),
        "args": vars(args),
        "full_cleaned_feature_count": len(full_cols),
        "transition_feature_count": len(trans_cols),
        "pca_probe": probes,
        "note": "All selection and fusion alpha choices are based on train/dev only; all_samples is external evaluation only.",
    }, out_dir / "transition_optimized_manifest.json")
    print(f"Wrote {out_dir / 'transition_optimized_summary.csv'}")


if __name__ == "__main__":
    main()
