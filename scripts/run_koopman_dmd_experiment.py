#!/usr/bin/env python3
"""Run Koopman-inspired DMD-lite spectral profiling experiments."""

from __future__ import annotations

import argparse
import json
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_transition_formal_experiment import (  # noqa: E402
    cleaned_full_columns,
    eval_one,
    fit_select,
    make_xy,
    merge_full_features,
    save_fig,
)
from run_transition_fullscale_optimized import build_base_datasets, composite_train_dev  # noqa: E402
from src.feature_koopman_dmd import (  # noqa: E402
    build_koopman_features,
    fit_loss_bins,
    load_loss_sequences,
    multiscale_features,
)
from src.train_eval import detector_metrics, get_importance, probabilities  # noqa: E402
from src.utils import write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cache_path(model_name: str, dataset_name: str) -> Path:
    return ROOT / "features_token_loss" / model_name / f"{dataset_name}_token_loss.jsonl.gz"


def feature_path(model_name: str, dataset_name: str) -> Path:
    return ROOT / "features_koopman" / model_name / f"{dataset_name}_koopman_features.csv"


def multiscale_path(dataset_name: str) -> Path:
    return ROOT / "features_koopman" / "multiscale" / f"{dataset_name}_koopman_multiscale_features.csv"


def base_dataset_names() -> list[str]:
    return [
        "m4_train",
        "m4_dev",
        "m4_test",
        "ghostbuster_train",
        "ghostbuster_dev",
        "ghostbuster_test",
        "hc3_plus_train",
        "hc3_plus_dev",
        "hc3_plus_test",
        "all_samples",
    ]


def all_dataset_names() -> list[str]:
    return base_dataset_names() + ["combined_strict_train", "combined_strict_dev", "leave_out_ghostbuster_train", "leave_out_ghostbuster_dev"]


def collect_train_sequences(data: dict[str, pd.DataFrame], model_name: str) -> list[np.ndarray]:
    caches = {name: load_loss_sequences(cache_path(model_name, name)) for name in base_dataset_names() if cache_path(model_name, name).exists()}
    seqs = []
    train_df, _ = composite_train_dev(data, "combined_strict")
    for rid in train_df["id"].astype(str):
        for cache in caches.values():
            if rid in cache:
                seqs.append(cache[rid])
                break
    return seqs


def ensure_koopman_features(data: dict[str, pd.DataFrame], models: list[str], force: bool = False) -> dict:
    manifest: dict[str, object] = {"models": {}}
    for model_name in models:
        seqs = collect_train_sequences(data, model_name)
        bins = fit_loss_bins(seqs, [5, 7])
        bins_path = ROOT / "features_koopman" / model_name / "public_train_loss_state_bins.json"
        save_json({"created_at": now(), "model_name": model_name, "bins_source": "combined_strict_train", "bins": bins}, bins_path)
        model_info = {"bins_path": str(bins_path), "datasets": {}}
        for name in base_dataset_names():
            cp = cache_path(model_name, name)
            if not cp.exists():
                model_info["datasets"][name] = {"status": "missing_cache", "cache": str(cp)}
                continue
            out = feature_path(model_name, name)
            regenerate = force or not out.exists()
            if not regenerate:
                try:
                    regenerate = len(pd.read_csv(out, usecols=["id"])) != len(data[name])
                except Exception:
                    regenerate = True
            if regenerate:
                build_koopman_features(cp, out, model_name=model_name, bins=bins)
            df = pd.read_csv(out, nrows=5)
            model_info["datasets"][name] = {"status": "ok", "path": str(out), "n_features": int(len(df.columns) - 1)}
        manifest["models"][model_name] = model_info
    if {"qwen25_1_5b", "qwen25_7b"}.issubset(set(models)):
        ms_info = {}
        for name in base_dataset_names():
            p15 = feature_path("qwen25_1_5b", name)
            p7 = feature_path("qwen25_7b", name)
            if p15.exists() and p7.exists():
                out = multiscale_path(name)
                regenerate = force or not out.exists()
                if not regenerate:
                    try:
                        existing = pd.read_csv(out, nrows=1)
                        regenerate = len(pd.read_csv(out, usecols=["id"])) != len(data[name]) or len(existing.columns) <= 1
                    except Exception:
                        regenerate = True
                if regenerate:
                    multiscale_features(pd.read_csv(p15), pd.read_csv(p7), out)
                ms_info[name] = {"status": "ok", "path": str(out), "n_features": int(len(pd.read_csv(out, nrows=1).columns) - 1)}
        manifest["multiscale"] = ms_info
    return manifest


def feature_pool(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame({"id": []})
    return pd.concat(frames, ignore_index=True).drop_duplicates("id")


def koopman_merge(meta: pd.DataFrame, model_names: list[str], include_multiscale: bool) -> pd.DataFrame:
    pools = []
    for model in model_names:
        pools.append(feature_pool([feature_path(model, name) for name in base_dataset_names()]))
    if include_multiscale:
        pools.append(feature_pool([multiscale_path(name) for name in base_dataset_names()]))
    out = meta.copy()
    for pool in pools:
        if len(pool.columns) > 1:
            out = out.merge(pool, on="id", how="left")
    return out


def rename_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return df.rename(columns={c: f"{c}_{suffix}" for c in df.columns if c != "id"})


def transition_file(model: str, train_name: str, dataset_name: str) -> Path | None:
    if model == "qwen25_1_5b":
        candidates = [
            ROOT / "features_transition" / "fullscale_1_5b_optimized" / model / train_name / f"{dataset_name}_transition_features.csv",
            ROOT / "features_transition" / "formal" / model / train_name / f"{dataset_name}_{model}_transition_features.csv",
            ROOT / "features_transition" / "formal" / model / train_name / f"{dataset_name}_transition_features.csv",
        ]
    else:
        candidates = [
            ROOT / "features_transition" / "formal" / model / train_name / f"{dataset_name}_transition_features.csv",
            ROOT / "features_transition" / "formal" / model / train_name / f"{dataset_name}_{model}_transition_features.csv",
        ]
    for path in candidates:
        if path.exists():
            return path
    return None


def merge_transition(meta: pd.DataFrame, train_name: str) -> pd.DataFrame:
    out = meta.copy()
    dataset_names = set(meta["transition_split"].astype(str)) if "transition_split" in meta.columns else set()
    pools = []
    for model, suffix in [("qwen25_1_5b", "one5"), ("qwen25_7b", "seven")]:
        frames = []
        for name in base_dataset_names() + [f"{train_name}_train", f"{train_name}_dev"]:
            path = transition_file(model, train_name, name)
            if path:
                frames.append(rename_cols(pd.read_csv(path), suffix))
        if frames:
            pools.append(pd.concat(frames, ignore_index=True).drop_duplicates("id"))
    for pool in pools:
        out = out.merge(pool, on="id", how="left")
    return out


def rank_metrics(m: dict) -> tuple[float, float, float, float]:
    return tuple(float(m.get(k, -np.inf)) if pd.notna(m.get(k, np.nan)) else -np.inf for k in ["auprc", "auroc", "tpr_at_fpr_5pct", "f1"])


def select_cols(train_df: pd.DataFrame, dev_df: pd.DataFrame, base_cols: list[str], candidate_cols: list[str], mode: str, seed: int) -> list[str]:
    candidate_cols = [c for c in candidate_cols if c in train_df.columns]
    if not candidate_cols:
        return base_cols
    if mode == "all":
        return base_cols + candidate_cols
    x_train, y_train, med = make_xy(train_df, candidate_cols)
    if mode == "l1":
        clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", penalty="l1", solver="liblinear", C=0.1))])
        clf.fit(x_train, y_train)
        coef = np.abs(clf.named_steps["model"].coef_[0])
        selected = [c for c, v in zip(candidate_cols, coef) if v > 1e-9]
        return base_cols + (selected or candidate_cols[: min(50, len(candidate_cols))])
    n = int(mode.replace("top", ""))
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", max_features="sqrt", n_jobs=-1)
    rf.fit(x_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1][: min(n, len(candidate_cols))]
    return base_cols + [candidate_cols[i] for i in order]


def choose_feature_selection(train_df: pd.DataFrame, dev_df: pd.DataFrame, base_cols: list[str], koop_cols: list[str], seed: int) -> tuple[str, list[str], pd.DataFrame]:
    rows = []
    best = None
    for mode in ["all", "top20", "top50", "top100", "l1"]:
        cols = select_cols(train_df, dev_df, base_cols, koop_cols, mode, seed)
        name, model, med, dev_metrics = fit_select(train_df, dev_df, cols)
        x_dev, y_dev, _ = make_xy(dev_df, cols, med)
        prob = probabilities(model, x_dev)
        m = detector_metrics(y_dev, prob)
        row = {"selection_mode": mode, "best_model": name, "n_features": len(cols), "n_koopman_features": len([c for c in cols if c in koop_cols]), **m}
        rows.append(row)
        if best is None or rank_metrics(m) > best[0]:
            best = (rank_metrics(m), mode, cols)
    return best[1], best[2], pd.DataFrame(rows)


def threshold_for_fpr(y_true, prob, target_fpr: float) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    thresholds = np.unique(np.r_[0.0, 1.0, p])
    best = (1.0, 0.0, 0.0)
    for thr in thresholds:
        pred = (p >= thr).astype(int)
        m = detector_metrics(y, p, y_pred=pred, threshold=float(thr))
        if m["fpr"] <= target_fpr + 1e-12 and m["tpr"] >= best[1]:
            best = (float(thr), float(m["tpr"]), float(m["fpr"]))
    return best


def eval_threshold(y_true, prob, thr: float) -> dict:
    pred = (np.asarray(prob) >= thr).astype(int)
    return detector_metrics(y_true, prob, y_pred=pred, threshold=thr)


def plot_outputs(summary: pd.DataFrame, all_samples: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not all_samples.empty:
        for metrics, stem, title in [
            (["auroc", "auprc", "f1"], "all_samples_performance", "Koopman/DMD-lite feature sets on all_samples"),
            (["tpr_at_fpr_1pct", "tpr_at_fpr_5pct"], "low_fpr_metrics", "Low-FPR performance on all_samples"),
        ]:
            long = all_samples[["train_name", "feature_set", *metrics]].melt(["train_name", "feature_set"], var_name="metric", value_name="value")
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.barplot(data=long, x="train_name", y="value", hue="feature_set", ax=ax)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=20)
            ax.legend(fontsize=7, ncol=2)
            save_fig(fig, plot_dir / stem)
        pivot = all_samples.pivot_table(index="train_name", columns="feature_set", values="auroc")
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("all_samples AUROC by train source and feature set")
        save_fig(fig, plot_dir / "feature_set_ablation_heatmap_auroc")
        pivot = all_samples.pivot_table(index="train_name", columns="feature_set", values="auprc")
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako", ax=ax)
        ax.set_title("all_samples AUPRC by train source and feature set")
        save_fig(fig, plot_dir / "feature_set_ablation_heatmap_auprc")
    best = all_samples.sort_values("auroc", ascending=False).head(12) if not all_samples.empty else pd.DataFrame()
    if not best.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        view = best[["train_name", "feature_set", "auroc", "auprc", "f1"]].melt(["train_name", "feature_set"], var_name="metric", value_name="value")
        view["model"] = view["train_name"] + " / " + view["feature_set"]
        sns.barplot(data=view, x="model", y="value", hue="metric", ax=ax)
        ax.set_title("Current best transition model vs Koopman variants")
        ax.tick_params(axis="x", rotation=35)
        save_fig(fig, plot_dir / "best_model_comparison")


def pca_probe(df: pd.DataFrame, koop_cols: list[str], plot_dir: Path) -> dict:
    if not koop_cols or len(df) < 10:
        return {}
    sample = df.groupby("transition_split", group_keys=False).apply(lambda g: g.sample(n=min(1000, len(g)), random_state=42)).reset_index(drop=True)
    x, _, med = make_xy(sample, koop_cols)
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(x))
    sample["PC1"] = coords[:, 0]
    sample["PC2"] = coords[:, 1]
    sample["label_name"] = sample["label"].map({0: "Human", 1: "AI"})
    for hue, stem in [("label_name", "koopman_pca_by_label"), ("source_dataset", "koopman_pca_by_source"), ("domain", "koopman_pca_by_domain")]:
        if hue in sample.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=sample, x="PC1", y="PC2", hue=hue, s=22, alpha=0.65, ax=ax)
            ax.set_title(f"Koopman feature PCA by {hue}")
            save_fig(fig, plot_dir / stem)
    probes = {}
    for label, col in [("label", "label"), ("source", "source_dataset"), ("domain", "domain")]:
        if col not in sample.columns or sample[col].nunique() < 2:
            continue
        y = sample[col].astype(str) if col != "label" else sample[col].astype(int)
        clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        clf.fit(x, y)
        probes[f"{label}_in_sample_probe_accuracy"] = float(clf.score(x, y))
    return probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--train_sources", nargs="+", default=["m4", "combined_strict", "leave_out_ghostbuster"])
    parser.add_argument("--test_sets", nargs="+", default=["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"])
    parser.add_argument("--models", nargs="+", default=["qwen25_1_5b", "qwen25_7b"])
    parser.add_argument("--run_koopman_features", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--run_feature_selection", action="store_true")
    parser.add_argument("--run_low_fpr_analysis", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_features", action="store_true")
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "results_koopman"
    ckpt_dir = ROOT / "checkpoints_koopman"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    if args.dry_run:
        missing = []
        for model in args.models:
            for name in base_dataset_names():
                if not cache_path(model, name).exists():
                    missing.append(str(cache_path(model, name)))
        print(json.dumps({"status": "dry_run_ok", "missing_caches": missing, "train_sources": args.train_sources, "test_sets": args.test_sets}, indent=2))
        return

    feature_manifest = ensure_koopman_features(data, args.models, force=args.force_features) if args.run_koopman_features else {}

    train_features = pd.read_csv(ROOT / args.full_features)
    ext_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)
    summary_rows = []
    selection_rows = []
    low_fpr_rows = []
    best_info = []
    pca_parts = []

    feature_sets = [
        "full_cleaned_only",
        "full_plus_transition_1_5b_7b",
        "koopman_only_1_5b",
        "koopman_only_7b",
        "koopman_only_1_5b_7b_multiscale",
        "full_plus_koopman_1_5b",
        "full_plus_koopman_7b",
        "full_plus_koopman_1_5b_7b",
        "full_plus_transition_plus_koopman",
    ]
    if not args.run_eval:
        save_json({"created_at": now(), "feature_manifest": feature_manifest, "note": "Features generated only."}, out_dir / "koopman_dmd_manifest.json")
        return

    for train_name in args.train_sources:
        train_meta, dev_meta = composite_train_dev(data, train_name)
        train_base = merge_full_features(train_meta, train_features, ext_features, full_cols)
        dev_base = merge_full_features(dev_meta, train_features, ext_features, full_cols)
        train_trans = merge_transition(train_base, train_name)
        dev_trans = merge_transition(dev_base, train_name)
        train_k15 = koopman_merge(train_base, ["qwen25_1_5b"], False)
        dev_k15 = koopman_merge(dev_base, ["qwen25_1_5b"], False)
        train_k7 = koopman_merge(train_base, ["qwen25_7b"], False)
        dev_k7 = koopman_merge(dev_base, ["qwen25_7b"], False)
        train_kall = koopman_merge(train_base, ["qwen25_1_5b", "qwen25_7b"], True)
        dev_kall = koopman_merge(dev_base, ["qwen25_1_5b", "qwen25_7b"], True)
        train_all = train_trans.merge(train_kall.drop(columns=[c for c in train_base.columns if c != "id"], errors="ignore"), on="id", how="left")
        dev_all = dev_trans.merge(dev_kall.drop(columns=[c for c in dev_base.columns if c != "id"], errors="ignore"), on="id", how="left")
        trans_cols = [c for c in train_trans.columns if c.endswith("_one5") or c.endswith("_seven")]
        k15_cols = [c for c in train_k15.columns if c.startswith("koopman_qwen25_1_5b_")]
        k7_cols = [c for c in train_k7.columns if c.startswith("koopman_qwen25_7b_")]
        kms_cols = [c for c in train_kall.columns if c.startswith("koopman_scale_")]
        kall_cols = k15_cols + k7_cols + kms_cols
        pca_parts.append(train_kall[["id", "label", "source_dataset", "domain", "transition_split", *kall_cols]].copy())
        set_frames = {
            "full_cleaned_only": (train_base, dev_base, full_cols, []),
            "full_plus_transition_1_5b_7b": (train_trans, dev_trans, full_cols + trans_cols, []),
            "koopman_only_1_5b": (train_k15, dev_k15, [], k15_cols),
            "koopman_only_7b": (train_k7, dev_k7, [], k7_cols),
            "koopman_only_1_5b_7b_multiscale": (train_kall, dev_kall, [], kall_cols),
            "full_plus_koopman_1_5b": (train_k15, dev_k15, full_cols, k15_cols),
            "full_plus_koopman_7b": (train_k7, dev_k7, full_cols, k7_cols),
            "full_plus_koopman_1_5b_7b": (train_kall, dev_kall, full_cols, kall_cols),
            "full_plus_transition_plus_koopman": (train_all, dev_all, full_cols + trans_cols, kall_cols),
        }
        fitted = {}
        for feature_set in feature_sets:
            tr, dv, base_cols, koop_cols = set_frames[feature_set]
            if koop_cols and args.run_feature_selection:
                mode, cols, sel_df = choose_feature_selection(tr, dv, base_cols, koop_cols, args.seed)
                sel_df.insert(0, "train_name", train_name)
                sel_df.insert(1, "feature_set", feature_set)
                selection_rows.extend(sel_df.to_dict("records"))
            else:
                mode, cols = "none", base_cols
            cols = [c for c in cols if c in tr.columns]
            if not cols:
                continue
            best_name, model, med, dev_candidates = fit_select(tr, dv, cols)
            x_dev, y_dev, _ = make_xy(dv, cols, med)
            dev_prob = probabilities(model, x_dev)
            ckpt = ckpt_dir / f"{train_name}_{feature_set}"
            ckpt.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, ckpt / "best_model.joblib")
            save_json(cols, ckpt / "feature_columns.json")
            save_json(med, ckpt / "feature_medians.json")
            save_json({"train_name": train_name, "feature_set": feature_set, "best_model": best_name, "selection_mode": mode, "created_at": now()}, ckpt / "train_metadata.json")
            fitted[feature_set] = (tr, cols, model, med, dev_prob, y_dev.to_numpy(dtype=int), best_name, mode)
            for test_name in args.test_sets:
                test_meta = data[test_name]
                test_base = merge_full_features(test_meta, train_features, ext_features, full_cols)
                if feature_set == "full_cleaned_only":
                    test_df = test_base
                elif feature_set == "full_plus_transition_1_5b_7b":
                    test_df = merge_transition(test_base, train_name)
                elif feature_set in {"koopman_only_1_5b", "full_plus_koopman_1_5b"}:
                    test_df = koopman_merge(test_base, ["qwen25_1_5b"], False)
                elif feature_set in {"koopman_only_7b", "full_plus_koopman_7b"}:
                    test_df = koopman_merge(test_base, ["qwen25_7b"], False)
                elif feature_set in {"koopman_only_1_5b_7b_multiscale", "full_plus_koopman_1_5b_7b"}:
                    test_df = koopman_merge(test_base, ["qwen25_1_5b", "qwen25_7b"], True)
                else:
                    test_trans = merge_transition(test_base, train_name)
                    test_k = koopman_merge(test_base, ["qwen25_1_5b", "qwen25_7b"], True)
                    test_df = test_trans.merge(test_k.drop(columns=[c for c in test_base.columns if c != "id"], errors="ignore"), on="id", how="left")
                metrics = eval_one(model, test_df, cols, med, out_dir / f"{train_name}_{feature_set}_to_{test_name}")
                summary_rows.append({
                    "train_name": train_name,
                    "feature_set": feature_set,
                    "test_set": test_name,
                    "best_model": best_name,
                    "selection_mode": mode,
                    "n_train": len(tr),
                    "n_test": len(test_df),
                    "n_features": len(cols),
                    "n_koopman_features": len([c for c in cols if c.startswith("koopman_")]),
                    **metrics,
                })
                if args.run_low_fpr_analysis and test_name == "all_samples":
                    x_test, y_test, _ = make_xy(test_df, cols, med)
                    test_prob = probabilities(model, x_test)
                    for target in [0.01, 0.05]:
                        thr, dev_tpr, dev_fpr = threshold_for_fpr(y_dev, dev_prob, target)
                        tm = eval_threshold(y_test, test_prob, thr)
                        low_fpr_rows.append({
                            "train_name": train_name,
                            "feature_set": feature_set,
                            "target_dev_fpr": target,
                            "threshold": thr,
                            "dev_tpr": dev_tpr,
                            "dev_fpr": dev_fpr,
                            "all_samples_tpr": tm["tpr"],
                            "all_samples_fpr": tm["fpr"],
                            "all_samples_precision": tm["precision"],
                            "all_samples_recall": tm["recall"],
                            "all_samples_f1": tm["f1"],
                        })
            if feature_set in {"full_plus_koopman_1_5b_7b", "full_plus_transition_plus_koopman", "koopman_only_1_5b_7b_multiscale"}:
                try:
                    imp = get_importance(model, cols)
                    imp = imp[imp["feature"].str.startswith("koopman_")].head(40)
                    if not imp.empty:
                        write_csv(imp, out_dir / f"{train_name}_{feature_set}_koopman_feature_importance.csv")
                        fig, ax = plt.subplots(figsize=(10, max(4, len(imp) * 0.18)))
                        sns.barplot(data=imp.iloc[::-1], y="feature", x="importance", ax=ax)
                        ax.set_title(f"Top Koopman feature importance: {train_name} {feature_set}")
                        save_fig(fig, plot_dir / f"{train_name}_{feature_set}_koopman_feature_importance")
                except Exception:
                    pass
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, out_dir / "koopman_dmd_summary.csv")
    all_samples = summary[summary["test_set"].eq("all_samples")].sort_values("auroc", ascending=False)
    write_csv(all_samples, out_dir / "koopman_dmd_all_samples_summary.csv")
    write_csv(pd.DataFrame(selection_rows), out_dir / "koopman_feature_selection_summary.csv")
    write_csv(pd.DataFrame(low_fpr_rows), out_dir / "low_fpr_analysis.csv")
    current_best = {
        "model_version": "current_best_transition_reference",
        "train_name": "leave_out_ghostbuster",
        "feature_set": "full_plus_1_5b_and_7b_transition",
        "auroc": 0.6951,
        "auprc": 0.6592,
        "f1": 0.6799,
        "tpr_at_fpr_5pct": 0.0933,
        "expected_calibration_error": 0.1488,
        "brier_score": 0.2459,
    }
    best_rows = [current_best]
    if not all_samples.empty:
        br = all_samples.iloc[0].to_dict()
        br["model_version"] = "best_koopman_run"
        best_rows.append(br)
    write_csv(pd.DataFrame(best_rows), out_dir / "best_model_comparison.csv")
    plot_outputs(summary, all_samples, plot_dir)
    probe = pca_probe(pd.concat(pca_parts, ignore_index=True).drop_duplicates("id"), [c for c in pca_parts[0].columns if c.startswith("koopman_")] if pca_parts else [], plot_dir)
    save_json({
        "created_at": now(),
        "args": vars(args),
        "feature_manifest": feature_manifest,
        "pca_probe": probe,
        "current_best_reference": current_best,
    }, out_dir / "koopman_dmd_manifest.json")
    print(f"Wrote {out_dir / 'koopman_dmd_summary.csv'}")


if __name__ == "__main__":
    main()
