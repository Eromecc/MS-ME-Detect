#!/usr/bin/env python3
"""Strict mathematical Text-Koopman experiment runner."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_transition_formal_experiment import cleaned_full_columns, eval_one, merge_full_features, save_fig  # noqa: E402
from run_transition_fullscale_optimized import build_base_datasets, composite_train_dev, test_df_for  # noqa: E402
from src import config  # noqa: E402
from src.feature_probability import load_causal_lm  # noqa: E402
from src.hidden_state_cache import cached_hidden_ids, extract_hidden_state_cache, read_hidden_manifest  # noqa: E402
from src.text_koopman_strict_math_audit import BANNED_FEATURE_TERMS, audit_strict_math_requirements  # noqa: E402
from src.text_koopman_strict_math_features import extract_strict_math_features  # noqa: E402
from src.text_koopman_strict_math_train import is_oom, train_strict_math_lifting  # noqa: E402
from src.train_eval import detector_metrics, probabilities  # noqa: E402
from src.utils import write_csv  # noqa: E402


sns.set_theme(style="whitegrid", context="talk")

BASE_DATASETS = ["m4_train", "m4_dev", "m4_test", "ghostbuster_train", "ghostbuster_dev", "ghostbuster_test", "hc3_plus_train", "hc3_plus_dev", "hc3_plus_test", "all_samples"]
TEST_SETS = ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]
PREVIOUS_BEST = {"auroc": 0.6951, "auprc": 0.6592, "tpr_at_fpr_5pct": 0.0933}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_names_needed(train_sources: list[str], test_sets: list[str]) -> list[str]:
    needed = set(test_sets)
    for train_name in train_sources:
        if train_name == "m4":
            needed.update(["m4_train", "m4_dev"])
        elif train_name == "leave_out_ghostbuster":
            needed.update(["m4_train", "m4_dev", "hc3_plus_train", "hc3_plus_dev"])
        elif train_name == "combined_strict":
            needed.update(["ghostbuster_train", "ghostbuster_dev", "m4_train", "m4_dev", "hc3_plus_train", "hc3_plus_dev"])
        else:
            raise ValueError(f"Unsupported train source: {train_name}")
    needed.update(BASE_DATASETS)
    return [x for x in BASE_DATASETS if x in needed]


def model_local_path(model_name: str) -> str:
    if model_name != "qwen25_1_5b":
        raise ValueError("Strict mathematical run is restricted to qwen25_1_5b; qwen25_14b is forbidden.")
    local = config.get_model_local_path("small")
    if not config.is_local_model_ready(local):
        raise FileNotFoundError(f"Local qwen25_1_5b model not ready at {local}; refusing to download.")
    return local


def load_qwen_for_hidden(model_name: str):
    return load_causal_lm(model_local_path(model_name), dtype=config.DTYPE, device_map=None, local_files_only=True)


def experiment_name(train_source: str, model_name: str, observable_dim: int, dmd_rank: int, max_length: int, max_rows_per_split: int | None) -> str:
    suffix = f"_n{max_rows_per_split}" if max_rows_per_split is not None else ""
    return f"{train_source}_{model_name}_hidden_to_obs{observable_dim}_rank{dmd_rank}_len{max_length}{suffix}"


def strict_feature_csv(feature_root: Path, exp: str, dataset_name: str) -> Path:
    return feature_root / exp / f"{dataset_name}_strict_koopman_features.csv"


def feature_complete(path: Path, checkpoint_created_at: str | None) -> bool:
    manifest = path.with_name(path.stem + "_manifest.json")
    if not path.exists() or not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return int(payload.get("n_features", 0)) > 0 and payload.get("checkpoint_created_at") == checkpoint_created_at
    except Exception:
        return False


def pool_features(feature_root: Path, exp: str, dataset_names: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(strict_feature_csv(feature_root, exp, name)) for name in dataset_names if strict_feature_csv(feature_root, exp, name).exists()]
    if not frames:
        return pd.DataFrame({"id": []})
    out = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    out["id"] = out["id"].astype(str)
    return out


def add_features(meta: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["id"] = out["id"].astype(str)
    if len(pool.columns) > 1:
        out = out.merge(pool, on="id", how="left")
    return out


def strict_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("strict_koopman_") and pd.api.types.is_numeric_dtype(df[c])]


def transition_path(train_source: str, model_name: str, dataset_name: str) -> Path | None:
    roots = [
        ROOT / "features_transition" / "formal" / model_name / train_source,
        ROOT / "features_transition" / "fullscale_1_5b_optimized" / model_name / train_source,
    ]
    stems = [f"{dataset_name}_{model_name}_transition_features.csv", f"{dataset_name}_transition_features.csv"]
    for root in roots:
        for stem in stems:
            path = root / stem
            if path.exists():
                return path
    return None


def merge_transition_features(meta: pd.DataFrame, train_source: str, model_name: str) -> pd.DataFrame:
    frames = []
    for ds in BASE_DATASETS + [f"{train_source}_train", f"{train_source}_dev"]:
        path = transition_path(train_source, model_name, ds)
        if path is not None:
            frames.append(pd.read_csv(path))
    if not frames:
        return meta.copy()
    pool = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    pool["id"] = pool["id"].astype(str)
    out = meta.copy()
    out["id"] = out["id"].astype(str)
    return out.merge(pool, on="id", how="left")


def transition_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        lc = col.lower()
        if col == "id" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.startswith("transition_") or "_transition_" in lc or "_trans_" in lc or lc.endswith("_transition_entropy"):
            out.append(col)
    return out


def rank_tuple(metrics: dict) -> tuple[float, float, float, float]:
    return tuple(float(metrics.get(k, -np.inf)) if pd.notna(metrics.get(k, np.nan)) else -np.inf for k in ["auprc", "auroc", "tpr_at_fpr_5pct", "f1"])


def classifier_candidates(seed: int) -> dict:
    return {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear"))]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1),
    }


def fit_select_classical(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str], *, seed: int):
    med = train_df[cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    x_train = train_df[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y_train = train_df["label"].astype(int)
    x_dev = dev_df[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y_dev = dev_df["label"].astype(int)
    rows = []
    best = None
    for name, clf in classifier_candidates(seed).items():
        clf.fit(x_train, y_train)
        prob = probabilities(clf, x_dev)
        pred = (prob >= 0.5).astype(int)
        metrics = detector_metrics(y_dev, prob, y_pred=pred)
        rows.append({"model": name, **metrics})
        rank = rank_tuple(metrics)
        if best is None or rank > best[0]:
            best = (rank, name, clf)
    if best is None:
        raise RuntimeError("No downstream classifier could be fit.")
    return best[1], best[2], med, pd.DataFrame(rows)


def ensure_hidden_caches(args, data: dict[str, pd.DataFrame], needed: list[str]) -> dict:
    def manifest_cached_ids(dataset_name: str) -> set[str]:
        manifest = read_hidden_manifest(ROOT / args.hidden_dir, args.model, dataset_name)
        ids = set(map(str, manifest.get("ids", [])))
        if ids and int(manifest.get("max_length", -1)) == int(args.max_length):
            return ids
        return cached_hidden_ids(ROOT / args.hidden_dir, args.model, dataset_name, max_length=args.max_length)

    info = {}
    model_name = args.model
    missing = []
    for name in needed:
        required = set(data[name]["id"].astype(str))
        cached = manifest_cached_ids(name)
        if not required.issubset(cached):
            missing.append({"dataset": name, "required": len(required), "cached": len(required & cached), "missing": len(required - cached)})
    if missing and not args.run_hidden_cache:
        return {"status": "missing_or_incomplete", "missing": missing}
    tokenizer = model = None
    if missing:
        tokenizer, model = load_qwen_for_hidden(model_name)
    for name in needed:
        required = set(data[name]["id"].astype(str))
        cached = manifest_cached_ids(name)
        if required.issubset(cached):
            info[name] = {"status": "cached_complete", "manifest": read_hidden_manifest(ROOT / args.hidden_dir, model_name, name)}
            continue
        info[name] = extract_hidden_state_cache(
            data[name],
            dataset_name=name,
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            output_root=ROOT / args.hidden_dir,
            max_length=args.max_length,
            batch_size=1,
            resume=True,
        )
    return {"status": "ok", "datasets": info}


def plot_outputs(summary: pd.DataFrame, plot_dir: Path) -> None:
    ok = summary[summary["status"].eq("ok")].copy() if not summary.empty and "status" in summary else pd.DataFrame()
    if ok.empty:
        return
    all_s = ok[ok["test_set"].eq("all_samples")]
    if not all_s.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=all_s, x="train_source", y="auroc", hue="feature_set", ax=ax)
        ax.set_title("Strict mathematical Text-Koopman on all_samples")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, plot_dir / "strict_math_vs_transition_barplot")
        fig, ax = plt.subplots(figsize=(10, 5))
        low = all_s.melt(id_vars=["train_source", "feature_set"], value_vars=["tpr_at_fpr_1pct", "tpr_at_fpr_5pct"], var_name="metric", value_name="value")
        sns.barplot(data=low, x="train_source", y="value", hue="feature_set", ax=ax)
        ax.set_title("Strict mathematical low-FPR comparison")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, plot_dir / "strict_math_low_fpr_comparison")
    spec = ok[ok["feature_set"].eq("strict_koopman_spectral_only")]
    if not spec.empty:
        pivot = spec.pivot_table(index="train_source", columns="test_set", values="auroc")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Strict mathematical source matrix AUROC")
        save_fig(fig, plot_dir / "strict_math_source_matrix_heatmap")


def plot_feature_distributions(df: pd.DataFrame, cols: list[str], plot_dir: Path) -> None:
    if df.empty or "label" not in df:
        return
    p = df.copy()
    p["label_name"] = p["label"].map({0: "Human", 1: "AI"})
    if "strict_koopman_spectral_radius" in p.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=p, x="strict_koopman_spectral_radius", hue="label_name", common_norm=False, ax=ax)
        ax.set_title("Strict spectral radius by label")
        save_fig(fig, plot_dir / "spectral_radius_by_label")
    if "strict_koopman_one_step_dmd_mse" in p.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=p, x="strict_koopman_one_step_dmd_mse", hue="label_name", common_norm=False, ax=ax)
        ax.set_title("Strict DMD residual by label")
        save_fig(fig, plot_dir / "dmd_residual_by_label")
    if {"strict_koopman_spectral_radius", "strict_koopman_eig_angle_mean"}.issubset(p.columns):
        q = p.copy()
        r = q["strict_koopman_spectral_radius"].replace([np.inf, -np.inf], np.nan)
        theta = q["strict_koopman_eig_angle_mean"].replace([np.inf, -np.inf], np.nan)
        q["eig_proxy_real"] = r * np.cos(theta)
        q["eig_proxy_imag"] = r * np.sin(theta)
        q = q[np.isfinite(q["eig_proxy_real"]) & np.isfinite(q["eig_proxy_imag"])]
        if not q.empty:
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.scatterplot(data=q, x="eig_proxy_real", y="eig_proxy_imag", hue="label_name", s=25, alpha=0.65, ax=ax)
            ax.add_patch(plt.Circle((0, 0), 1.0, color="black", fill=False, linestyle="--", linewidth=1.0))
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Strict local-K eigenvalue complex-plane proxy")
            save_fig(fig, plot_dir / "eigenvalue_complex_plane_examples")


def write_report(result_dir: Path, summary: pd.DataFrame, audit: dict, meta: dict | None) -> None:
    all_s = summary[summary["test_set"].eq("all_samples")] if not summary.empty and "test_set" in summary else pd.DataFrame()
    best_combo = all_s.sort_values(["auroc", "auprc"], ascending=False).head(1) if not all_s.empty else pd.DataFrame()
    exceeded = False
    low_fpr_improved = False
    if not best_combo.empty:
        row = best_combo.iloc[0]
        exceeded = bool(row.get("auroc", -np.inf) > PREVIOUS_BEST["auroc"] and row.get("auprc", -np.inf) > PREVIOUS_BEST["auprc"])
        low_fpr_improved = bool(row.get("tpr_at_fpr_5pct", -np.inf) > PREVIOUS_BEST["tpr_at_fpr_5pct"])
    lines = [
        "# Strict Mathematical Text-Koopman Report",
        "",
        f"Created at: {now()}",
        "",
        f"- audit_passed: `{audit.get('passed')}`",
        f"- hidden_size: `{None if meta is None else meta.get('hidden_size')}`",
        f"- observable_dim: `{None if meta is None else meta.get('observable_dim')}`",
        f"- uses_projection: `{None if meta is None else meta.get('uses_projection')}`",
        f"- uses_label_classifier: `{None if meta is None else meta.get('uses_label_classifier')}`",
        f"- uses_global_K_parameter: `{None if meta is None else meta.get('uses_global_K_parameter')}`",
        f"- strict_math_failed_due_to_memory: `{audit.get('strict_math_failed_due_to_memory')}`",
        f"- exceeded_previous_best_auroc_and_auprc: `{exceeded}`",
        f"- improved_tpr_at_fpr5: `{low_fpr_improved}`",
        "",
        "Strict mathematical Text-Koopman was implemented as specified, but under current data it does not outperform transition-state profiling unless the summary table proves otherwise.",
        "",
    ]
    if not all_s.empty:
        lines.extend(["## all_samples", "", "```text", all_s.to_string(index=False), "```", ""])
    (result_dir / "STRICT_TEXT_KOOPMAN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--hidden_dir", default="features_hidden_states")
    parser.add_argument("--feature_dir", default="features_text_koopman_strict_math")
    parser.add_argument("--checkpoint_dir", default="checkpoints_text_koopman_strict_math")
    parser.add_argument("--output_dir", default="results_text_koopman_strict_math")
    parser.add_argument("--train_sources", nargs="+", default=["leave_out_ghostbuster"])
    parser.add_argument("--test_sets", nargs="+", default=TEST_SETS)
    parser.add_argument("--model", default="qwen25_1_5b")
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--observable_multiplier", type=int, default=2)
    parser.add_argument("--observable_dim", type=int, default=None)
    parser.add_argument("--dmd_ranks", nargs="+", type=int, default=[32])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_hidden_cache", action="store_true")
    parser.add_argument("--run_lifting_train", action="store_true")
    parser.add_argument("--run_feature_extract", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--run_audit", action="store_true")
    parser.add_argument("--run_plots", action="store_true")
    args = parser.parse_args()

    if args.model != "qwen25_1_5b":
        raise ValueError("Only qwen25_1_5b is allowed for strict mathematical experiments.")
    data = build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    needed = dataset_names_needed(args.train_sources, args.test_sets)
    result_dir = ROOT / args.output_dir
    feature_root = ROOT / args.feature_dir
    ckpt_root = ROOT / args.checkpoint_dir
    plot_dir = result_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    first_manifest = read_hidden_manifest(ROOT / args.hidden_dir, args.model, "m4_train")
    hidden_size = int(first_manifest.get("hidden_size", 0)) if first_manifest else 0
    observable_dim = int(args.observable_dim or (args.observable_multiplier * hidden_size)) if hidden_size else None
    dry_info = {
        "created_at": now(),
        "train_sources": args.train_sources,
        "test_sets": args.test_sets,
        "model": args.model,
        "needed_datasets": {k: len(data[k]) for k in needed},
        "hidden_size_from_manifest": hidden_size or None,
        "observable_dim": observable_dim,
        "strict_policy": "hidden_states directly enter g_theta; no PCA/random projection; downstream strict_koopman_* spectral features only.",
    }
    if args.dry_run:
        save_json(dry_info, result_dir / "strict_math_dry_run.json")
        print(json.dumps(dry_info, indent=2))
        return

    cache_info = ensure_hidden_caches(args, data, needed)
    if cache_info.get("status") == "missing_or_incomplete":
        save_json({"created_at": now(), "strict_math_failed_due_to_memory": False, "cache_info": cache_info}, result_dir / "strict_math_manifest.json")
        raise FileNotFoundError(f"Hidden cache is incomplete and --run_hidden_cache was not set: {cache_info}")

    train_features = pd.read_csv(ROOT / args.full_features)
    external_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)
    summary_rows = []
    manifest = {"created_at": now(), "args": vars(args), "hidden_cache": cache_info, "experiments": {}, "errors": []}
    latest_meta = None
    latest_audit = {}

    for train_source in args.train_sources:
        train_meta, dev_meta = composite_train_dev(data, train_source)
        for dmd_rank in args.dmd_ranks:
            hidden_manifest = read_hidden_manifest(ROOT / args.hidden_dir, args.model, "m4_train")
            hidden_size = int(hidden_manifest.get("hidden_size", hidden_size))
            observable_dim = int(args.observable_dim or (args.observable_multiplier * hidden_size))
            exp = experiment_name(train_source, args.model, observable_dim, dmd_rank, args.max_length, args.max_rows_per_split)
            ckpt = ckpt_root / exp
            try:
                meta_path = ckpt / "strict_math_metadata.json"
                if not meta_path.exists():
                    if not args.run_lifting_train:
                        raise RuntimeError(f"Missing strict-math checkpoint: {ckpt}")
                    train_dataset_names = [name for name in needed if name.endswith("_train") or name.endswith("_dev")]
                    train_strict_math_lifting(
                        train_meta=train_meta,
                        dev_meta=dev_meta,
                        hidden_root=ROOT / args.hidden_dir,
                        model_name=args.model,
                        dataset_names=train_dataset_names,
                        output_dir=ckpt,
                        observable_dim=observable_dim,
                        observable_multiplier=args.observable_multiplier,
                        dmd_rank=dmd_rank,
                        max_seq_len=args.max_length,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        device=args.device,
                    )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                latest_meta = meta
                if args.run_feature_extract:
                    feature_device = "cuda" if args.device == "auto" and __import__("torch").cuda.is_available() else args.device
                    for ds in needed:
                        out_csv = strict_feature_csv(feature_root, exp, ds)
                        allowed = set(data[ds]["id"].astype(str)) if args.max_rows_per_split is not None else None
                        if feature_complete(out_csv, meta.get("created_at")):
                            continue
                        extract_strict_math_features(
                            hidden_root=ROOT / args.hidden_dir,
                            model_name=args.model,
                            dataset_name=ds,
                            checkpoint_dir=ckpt,
                            output_csv=out_csv,
                            dmd_rank=dmd_rank,
                            max_seq_len=args.max_length,
                            allowed_ids=allowed,
                            device=feature_device,
                        )
                pool = pool_features(feature_root, exp, needed)
                train_base = add_features(train_meta, pool)
                dev_base = add_features(dev_meta, pool)
                cols = strict_cols(train_base)
                if args.run_audit or args.run_eval:
                    latest_audit = audit_strict_math_requirements(
                        metadata=meta,
                        feature_columns=cols,
                        hidden_manifest=hidden_manifest,
                        output_dir=result_dir,
                        docs_dir=ROOT / "docs",
                        strict_math_failed_due_to_memory=bool(meta.get("strict_math_failed_due_to_memory")),
                    )
                    if not latest_audit["passed"]:
                        raise RuntimeError(f"Strict math audit failed: {latest_audit}")
                if not args.run_eval:
                    manifest["experiments"][exp] = {"status": "ok", "checkpoint": str(ckpt), "metadata": meta}
                    continue
                feature_sets = {"strict_koopman_spectral_only": cols}
                full_train = merge_full_features(train_meta, train_features, external_features, full_cols)
                full_dev = merge_full_features(dev_meta, train_features, external_features, full_cols)
                train_full_strict = full_train.merge(train_base[["id", *cols]], on="id", how="left")
                dev_full_strict = full_dev.merge(dev_base[["id", *cols]], on="id", how="left")
                feature_sets["full_plus_strict_koopman"] = full_cols + cols
                trans_train = merge_transition_features(train_base, train_source, args.model)
                trans_dev = merge_transition_features(dev_base, train_source, args.model)
                trans_cols = transition_cols(trans_train)
                if trans_cols:
                    feature_sets["transition_plus_strict_koopman"] = trans_cols + cols
                    train_full_trans_strict = merge_transition_features(train_full_strict, train_source, args.model)
                    dev_full_trans_strict = merge_transition_features(dev_full_strict, train_source, args.model)
                    feature_sets["full_plus_transition_plus_strict_koopman"] = full_cols + trans_cols + cols
                else:
                    train_full_trans_strict = train_full_strict
                    dev_full_trans_strict = dev_full_strict
                train_map = {
                    "strict_koopman_spectral_only": train_base,
                    "full_plus_strict_koopman": train_full_strict,
                    "transition_plus_strict_koopman": trans_train,
                    "full_plus_transition_plus_strict_koopman": train_full_trans_strict,
                }
                dev_map = {
                    "strict_koopman_spectral_only": dev_base,
                    "full_plus_strict_koopman": dev_full_strict,
                    "transition_plus_strict_koopman": trans_dev,
                    "full_plus_transition_plus_strict_koopman": dev_full_trans_strict,
                }
                for fs, fs_cols in feature_sets.items():
                    if not fs_cols:
                        continue
                    bad = [c for c in fs_cols if any(term in c.lower() for term in BANNED_FEATURE_TERMS)]
                    if bad:
                        raise RuntimeError(f"Banned downstream feature names: {bad}")
                    best_name, best_model, med, selection_df = fit_select_classical(train_map[fs], dev_map[fs], fs_cols, seed=args.seed)
                    selection_path = result_dir / f"{exp}_{fs}_classifier_selection.csv"
                    write_csv(selection_df, selection_path)
                    joblib.dump({"model": best_model, "medians": med, "columns": fs_cols, "feature_set": fs}, ckpt / f"{fs}_classifier.joblib")
                    for test_name in args.test_sets:
                        meta_df = test_df_for(data, test_name)
                        test_base = add_features(meta_df, pool)
                        test_df = test_base
                        if fs.startswith("full_plus"):
                            test_df = merge_full_features(meta_df, train_features, external_features, full_cols).merge(test_base[["id", *cols]], on="id", how="left")
                        if "transition" in fs:
                            test_df = merge_transition_features(test_df, train_source, args.model)
                        out = result_dir / f"{exp}_{fs}_to_{test_name}"
                        metrics = eval_one(best_model, test_df, fs_cols, med, out)
                        write_csv(pd.DataFrame([metrics]), out / "detector_metrics.csv")
                        summary_rows.append({"status": "ok", "experiment": exp, "model": args.model, "train_source": train_source, "feature_set": fs, "test_set": test_name, "best_model": best_name, "n_features": len(fs_cols), **metrics})
                manifest["experiments"][exp] = {"status": "ok", "checkpoint": str(ckpt), "metadata": meta}
                plot_feature_distributions(pd.concat([dev_base, add_features(data["all_samples"], pool)], ignore_index=True), cols, plot_dir)
            except RuntimeError as exc:
                failed_due_memory = is_oom(exc)
                err = {"experiment": exp, "error": repr(exc), "strict_math_failed_due_to_memory": failed_due_memory}
                manifest["errors"].append(err)
                save_json(err, result_dir / f"{exp}_failure.json")
                if failed_due_memory:
                    continue
                continue
            except Exception as exc:
                manifest["errors"].append({"experiment": exp, "error": repr(exc), "strict_math_failed_due_to_memory": False})
                continue

    summary = pd.DataFrame(summary_rows)
    write_csv(summary, result_dir / "strict_math_summary.csv")
    if not summary.empty:
        write_csv(summary[summary["test_set"].eq("all_samples")], result_dir / "strict_math_all_samples_summary.csv")
        prev = pd.DataFrame([{"model_version": "previous_best_transition", "train_source": "leave_out_ghostbuster", "feature_set": "full_plus_1_5b_and_7b_transition", **PREVIOUS_BEST}])
        best = summary[summary["test_set"].eq("all_samples")].sort_values(["auroc", "auprc"], ascending=False).head(10)
        write_csv(pd.concat([prev, best], ignore_index=True), result_dir / "strict_math_vs_previous_best.csv")
        source = summary[["train_source", "test_set", "feature_set", "auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct"]].copy()
        write_csv(source, result_dir / "strict_math_source_matrix.csv")
        if args.run_plots:
            plot_outputs(summary, plot_dir)
    write_report(result_dir, summary, latest_audit, latest_meta)
    manifest["completed_at"] = now()
    manifest["summary_rows"] = int(len(summary_rows))
    save_json(manifest, result_dir / "strict_math_manifest.json")


if __name__ == "__main__":
    main()
