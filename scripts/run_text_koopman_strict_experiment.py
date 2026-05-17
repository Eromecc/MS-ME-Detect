#!/usr/bin/env python3
"""Strict Text-Koopman spectral-only experiment runner.

Pipeline:
hidden trajectory -> learned lifting g_theta -> per-document local K_i ->
spectral/residual scalar features -> classical classifier.
"""

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
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_transition_formal_experiment import cleaned_full_columns, eval_one, fit_select, merge_full_features, save_fig  # noqa: E402
from run_transition_fullscale_optimized import build_base_datasets, composite_train_dev, test_df_for  # noqa: E402
from src import config  # noqa: E402
from src.feature_probability import load_causal_lm  # noqa: E402
from src.hidden_state_cache import cached_hidden_ids, extract_hidden_state_cache, read_hidden_manifest, rebuild_hidden_manifest_from_shards  # noqa: E402
from src.text_koopman_features import extract_text_koopman_features, load_feature_artifacts  # noqa: E402
from src.text_koopman_train import train_text_koopman_lifting  # noqa: E402
from src.train_eval import detector_metrics, probabilities  # noqa: E402
from src.utils import write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


BASE_DATASETS = ["m4_train", "m4_dev", "m4_test", "ghostbuster_train", "ghostbuster_dev", "ghostbuster_test", "hc3_plus_train", "hc3_plus_dev", "hc3_plus_test", "all_samples"]
TEST_SETS = ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]
MODEL_KEYS = {"qwen25_1_5b": "small", "qwen25_7b": "medium"}
BANNED_FEATURE_TERMS = ["pooled", "hidden_mean", "hidden_std", "z_mean", "z_std", "cls", "token_id", "token_text", "embedding_mean"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_names_needed(train_sources: list[str], test_sets: list[str]) -> list[str]:
    # Cache/extract all public splits for audit completeness.  The downstream
    # train/dev composites still enforce the requested training-source policy,
    # and all_samples remains external-test only.
    needed = set()
    for train_name in train_sources:
        if train_name == "m4":
            needed.update(["m4_train", "m4_dev"])
        elif train_name == "leave_out_ghostbuster":
            needed.update(["m4_train", "m4_dev", "hc3_plus_train", "hc3_plus_dev"])
        elif train_name == "combined_strict":
            needed.update(["ghostbuster_train", "ghostbuster_dev", "m4_train", "m4_dev", "hc3_plus_train", "hc3_plus_dev"])
        else:
            raise ValueError(f"Unsupported train source for targeted strict Text-Koopman: {train_name}")
    needed.update(test_sets)
    needed.update(BASE_DATASETS)
    return [x for x in BASE_DATASETS if x in needed]


def model_local_path(model_name: str) -> str:
    key = MODEL_KEYS.get(model_name)
    if key is None:
        raise ValueError(f"Unsupported model_name={model_name}; allowed: {sorted(MODEL_KEYS)}")
    local = config.get_model_local_path(key)
    if not config.is_local_model_ready(local):
        raise FileNotFoundError(f"Local model not ready at {local}; refusing to download.")
    return local


def load_qwen_for_hidden(model_name: str):
    return load_causal_lm(model_local_path(model_name), dtype=config.DTYPE, device_map=None, local_files_only=True)


def experiment_name(
    train_source: str,
    model_name: str,
    projection_dim: int,
    projector_type: str,
    observable_dim: int,
    dmd_rank: int,
    max_length: int,
    max_rows_per_split: int | None = None,
) -> str:
    suffix = f"_n{max_rows_per_split}" if max_rows_per_split is not None else ""
    projector_suffix = "" if projector_type == "random" else f"_{projector_type}"
    return f"{train_source}_{model_name}_proj{projection_dim}{projector_suffix}_obs{observable_dim}_rank{dmd_rank}_len{max_length}{suffix}"


def checkpoint_compatible(ckpt: Path, *, train_rows: int, dev_rows: int, projection_dim: int, observable_dim: int, dmd_rank: int, max_length: int, projector_type: str = "random") -> bool:
    meta_path = ckpt / "lifting_metadata.json"
    model_path = ckpt / "text_koopman_lifting.pt"
    if not meta_path.exists() or not model_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = {
        "projection_dim": int(projection_dim),
        "observable_dim": int(observable_dim),
        "dmd_rank": int(dmd_rank),
        "max_seq_len": int(max_length),
    }
    for key, val in expected.items():
        if int(meta.get(key, -1)) != val:
            return False
    if str(meta.get("projector_type", "random")) != str(projector_type):
        return False
    if int(meta.get("n_train_sequences_used", 0)) < int(train_rows):
        return False
    if int(meta.get("n_dev_sequences_used", 0)) < int(dev_rows):
        return False
    return True


def feature_csv(feature_dir: Path, exp: str, dataset_name: str) -> Path:
    return feature_dir / exp / f"{dataset_name}_text_koopman_features.csv"


def feature_csv_complete(path: Path, required_rows: int, checkpoint_created_at: str | None = None) -> bool:
    manifest = path.with_name(path.stem + "_manifest.json")
    if not path.exists() or not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if checkpoint_created_at is not None and payload.get("checkpoint_created_at") != checkpoint_created_at:
            return False
        # Some valid samples are skipped when the hidden trajectory is too
        # short for local DMD. Treat an existing manifest with feature columns
        # as complete for resume purposes; the skipped ids are recorded there.
        return int(payload.get("n_features", 0)) > 0 and int(payload.get("n_rows", 0)) > 0
    except Exception:
        return False


def pool_features(feature_dir: Path, exp: str, dataset_names: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(feature_csv(feature_dir, exp, name)) for name in dataset_names if feature_csv(feature_dir, exp, name).exists()]
    if not frames:
        return pd.DataFrame({"id": []})
    return pd.concat(frames, ignore_index=True).drop_duplicates("id")


def add_features(meta: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["id"] = out["id"].astype(str)
    if len(pool.columns) > 1:
        p = pool.copy()
        p["id"] = p["id"].astype(str)
        out = out.merge(p, on="id", how="left")
    return out


def text_koopman_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("text_koopman_") and pd.api.types.is_numeric_dtype(df[c])]


def transition_path(train_source: str, model_name: str, dataset_name: str) -> Path | None:
    roots = [
        ROOT / "features_transition" / "formal" / model_name / train_source,
        ROOT / "features_transition" / "fullscale_1_5b_optimized" / model_name / train_source,
    ]
    stems = [
        f"{dataset_name}_{model_name}_transition_features.csv",
        f"{dataset_name}_transition_features.csv",
    ]
    for root in roots:
        for stem in stems:
            p = root / stem
            if p.exists():
                return p
    return None


def merge_transition_features(meta: pd.DataFrame, train_source: str, model_names: list[str]) -> pd.DataFrame:
    frames = []
    for model_name in model_names:
        for ds in BASE_DATASETS + [f"{train_source}_train", f"{train_source}_dev"]:
            p = transition_path(train_source, model_name, ds)
            if p is not None:
                frames.append(pd.read_csv(p))
    if not frames:
        return meta.copy()
    pool = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    pool["id"] = pool["id"].astype(str)
    out = meta.copy()
    out["id"] = out["id"].astype(str)
    return out.merge(pool, on="id", how="left")


def transition_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if c == "id" or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith("transition_") or "_transition_" in lc or "_trans_" in lc or lc.endswith("_transition_entropy"):
            cols.append(c)
    return cols


def write_composite_features(feature_dir: Path, exp: str, train_source: str, data: dict[str, pd.DataFrame]) -> None:
    train_meta, dev_meta = composite_train_dev(data, train_source)
    pool = pool_features(feature_dir, exp, BASE_DATASETS)
    for name, meta in [(f"{train_source}_train", train_meta), (f"{train_source}_dev", dev_meta)]:
        out = add_features(meta[["id"]].copy(), pool)
        out.to_csv(feature_csv(feature_dir, exp, name), index=False)


def ensure_hidden_caches(args, data: dict[str, pd.DataFrame], needed: list[str]) -> dict:
    def ids_covering_required(model_name: str, dataset_name: str, required_ids: set[str]) -> set[str]:
        man = read_hidden_manifest(ROOT / args.hidden_dir, model_name, dataset_name)
        manifest_ids = set(map(str, man.get("ids", [])))
        if manifest_ids and int(man.get("max_length", -1)) == int(args.max_length) and required_ids.issubset(manifest_ids):
            return manifest_ids
        return cached_hidden_ids(ROOT / args.hidden_dir, model_name, dataset_name, max_length=args.max_length)

    info = {}
    for model_name in args.models:
        missing = []
        incomplete = {}
        for name in needed:
            required_ids = set(data[name]["id"].astype(str))
            cached_ids = ids_covering_required(model_name, name, required_ids)
            missing_ids = sorted(required_ids - cached_ids)
            if missing_ids:
                missing.append(name)
                incomplete[name] = {"required": len(required_ids), "cached_matching": len(required_ids & cached_ids), "missing": len(missing_ids)}
        if missing and not args.run_hidden_cache:
            info[model_name] = {"status": "missing_or_incomplete", "missing": missing, "incomplete": incomplete}
            continue
        tokenizer = model = None
        if missing and args.run_hidden_cache:
            tokenizer, model = load_qwen_for_hidden(model_name)
        model_info = {}
        for name in needed:
            man = read_hidden_manifest(ROOT / args.hidden_dir, model_name, name)
            required_ids = set(data[name]["id"].astype(str))
            cached_ids = ids_covering_required(model_name, name, required_ids)
            missing_ids = sorted(required_ids - cached_ids)
            if not missing_ids and not args.force_hidden_cache:
                if int(man.get("max_length", -1)) == int(args.max_length) and len(set(map(str, man.get("ids", [])))) != len(cached_ids):
                    man = rebuild_hidden_manifest_from_shards(ROOT / args.hidden_dir, model_name, name)
                model_info[name] = {"status": "cached_complete", "manifest": man}
                continue
            if tokenizer is None or model is None:
                model_info[name] = {"status": "missing_or_incomplete", "missing_ids": len(missing_ids)}
                continue
            model_info[name] = extract_hidden_state_cache(
                data[name],
                dataset_name=name,
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                output_root=ROOT / args.hidden_dir,
                max_length=args.max_length,
                batch_size=args.hidden_batch_size,
                resume=True,
            )
        info[model_name] = model_info
    return info


def rank_tuple(m: dict) -> tuple[float, float, float, float]:
    return tuple(float(m.get(k, -np.inf)) if pd.notna(m.get(k, np.nan)) else -np.inf for k in ["auprc", "auroc", "tpr_at_fpr_5pct", "f1"])


def fit_logistic_fast(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str]):
    med = train_df[cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    x_train = train_df[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y_train = train_df["label"].astype(int)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
        ]
    )
    clf.fit(x_train, y_train)
    return "LogisticRegression_fast", clf, med, {}


def classifier_candidates(seed: int) -> dict:
    out = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        out["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
        )
    except Exception:
        pass
    return out


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
        row = {"model": name, **metrics}
        rows.append(row)
        rank = rank_tuple(metrics)
        if best is None or rank > best[0]:
            best = (rank, name, clf)
    if best is None:
        raise RuntimeError("No downstream classifier candidate could be fit.")
    return best[1], best[2], med, pd.DataFrame(rows)


def plot_outputs(summary: pd.DataFrame, plot_dir: Path) -> None:
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        return
    all_s = ok[ok["test_set"].eq("all_samples")]
    if not all_s.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=all_s, x="train_source", y="auroc", hue="feature_set", ax=ax)
        ax.set_title("Strict Text-Koopman vs feature combinations on all_samples")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, plot_dir / "text_koopman_vs_transition_barplot")
        fig, ax = plt.subplots(figsize=(10, 5))
        low = all_s.melt(id_vars=["train_source", "feature_set"], value_vars=["tpr_at_fpr_1pct", "tpr_at_fpr_5pct"], var_name="metric", value_name="value")
        sns.barplot(data=low, x="train_source", y="value", hue="feature_set", ax=ax)
        ax.set_title("Strict Text-Koopman low-FPR comparison")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, plot_dir / "low_fpr_comparison")
    spec = ok[ok["feature_set"].eq("text_koopman_spectral_only")]
    if not spec.empty:
        pivot = spec.pivot_table(index="train_source", columns="test_set", values="auroc")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Strict Text-Koopman spectral-only AUROC")
        save_fig(fig, plot_dir / "text_koopman_source_matrix_auroc_heatmap")


def plot_transition_delta_heatmap(delta_path: Path, plot_dir: Path) -> None:
    if not delta_path.exists():
        return
    delta = pd.read_csv(delta_path)
    if delta.empty or "delta_auroc" not in delta.columns:
        return
    for metric, stem in [("delta_auroc", "text_koopman_delta_vs_transition_heatmap")]:
        pivot = delta.pivot_table(index="feature_set", columns="test_set", values=metric)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", center=0.0, cmap="coolwarm", ax=ax)
        ax.set_title("Strict Text-Koopman delta vs transition reference")
        save_fig(fig, plot_dir / stem)


def plot_spectral_pca(df: pd.DataFrame, cols: list[str], plot_dir: Path) -> None:
    if len(cols) < 2 or len(df) < 10:
        return
    x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median(numeric_only=True)).fillna(0.0)
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(x))
    p = df.copy()
    p["PC1"] = coords[:, 0]
    p["PC2"] = coords[:, 1]
    p["label_name"] = p["label"].map({0: "Human", 1: "AI"})
    for hue, stem in [("label_name", "pca_spectral_features_by_label"), ("source_dataset", "pca_spectral_features_by_source")]:
        if hue in p.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=p, x="PC1", y="PC2", hue=hue, s=25, alpha=0.7, ax=ax)
            ax.set_title(f"Strict Text-Koopman spectral PCA by {hue}")
            save_fig(fig, plot_dir / stem)
    if "text_koopman_spectral_radius" in p.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=p, x="text_koopman_spectral_radius", hue="label_name", common_norm=False, ax=ax)
        ax.set_title("Spectral radius distribution by label")
        save_fig(fig, plot_dir / "spectral_radius_distribution_by_label")
    if "text_koopman_one_step_dmd_mse" in p.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=p, x="text_koopman_one_step_dmd_mse", hue="label_name", common_norm=False, ax=ax)
        ax.set_title("DMD residual distribution by label")
        save_fig(fig, plot_dir / "dmd_residual_distribution_by_label")
    if {"text_koopman_spectral_radius", "text_koopman_eig_angle_mean"}.issubset(p.columns):
        r = p["text_koopman_spectral_radius"].replace([np.inf, -np.inf], np.nan)
        theta = p["text_koopman_eig_angle_mean"].replace([np.inf, -np.inf], np.nan)
        q = p.copy()
        q["eig_proxy_real"] = r * np.cos(theta)
        q["eig_proxy_imag"] = r * np.sin(theta)
        q = q[np.isfinite(q["eig_proxy_real"]) & np.isfinite(q["eig_proxy_imag"])].copy()
        if not q.empty:
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.scatterplot(data=q, x="eig_proxy_real", y="eig_proxy_imag", hue="label_name", s=25, alpha=0.65, ax=ax)
            circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linestyle="--", linewidth=1.0)
            ax.add_patch(circle)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Eigenvalue complex-plane proxy from local-K summaries")
            save_fig(fig, plot_dir / "eigenvalue_complex_plane_examples")


def run_probe(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    if not cols:
        return rows
    x_train = train_df[cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[cols].median(numeric_only=True)).fillna(0.0)
    x_dev = dev_df[cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[cols].median(numeric_only=True)).fillna(0.0)
    for name, col in [("label", "label"), ("source", "source_dataset"), ("domain", "domain")]:
        if col not in train_df.columns or train_df[col].nunique() < 2:
            continue
        y_train = train_df[col].astype(str) if col != "label" else train_df[col].astype(int)
        y_dev = dev_df[col].astype(str) if col != "label" else dev_df[col].astype(int)
        clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        clf.fit(x_train, y_train)
        rows.append({"probe": name, "dev_accuracy": float(clf.score(x_dev, y_dev)), "n_features": len(cols)})
    return rows


def _single_feature_probe_score(train_df: pd.DataFrame, dev_df: pd.DataFrame, feature: str, target: str) -> float:
    if target not in train_df.columns or train_df[target].nunique() < 2 or dev_df[target].nunique() < 2:
        return np.nan
    med = train_df[[feature]].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    x_train = train_df[[feature]].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_dev = dev_df[[feature]].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y_train = train_df[target].astype(str) if target != "label" else train_df[target].astype(int)
    y_dev = dev_df[target].astype(str) if target != "label" else dev_df[target].astype(int)
    clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_dev)
    return float(balanced_accuracy_score(y_dev, pred))


def source_guarded_spectral_cols(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    cols: list[str],
    *,
    max_source_acc: float = 0.75,
    source_margin: float = 0.02,
    min_features: int = 12,
    mode: str = "univariate",
) -> tuple[list[str], pd.DataFrame]:
    """Filter spectral scalar features using public train/dev probes only.

    The guard is intentionally conservative: a feature is kept when its
    univariate source probe is not above the configured ceiling or when its
    label probe is at least competitive with its source probe. If this would
    leave too few features, the least source-dominant features are retained.
    """
    rows = []
    for col in cols:
        label_acc = _single_feature_probe_score(train_df, dev_df, col, "label")
        source_acc = _single_feature_probe_score(train_df, dev_df, col, "source_dataset")
        source_minus_label = source_acc - label_acc if pd.notna(source_acc) and pd.notna(label_acc) else np.nan
        keep = pd.isna(source_acc) or source_acc <= max_source_acc or source_minus_label <= source_margin
        rows.append(
            {
                "feature": col,
                "label_probe_balanced_accuracy": label_acc,
                "source_probe_balanced_accuracy": source_acc,
                "source_minus_label": source_minus_label,
                "kept": bool(keep),
            }
        )
    report = pd.DataFrame(rows)
    kept = report[report["kept"]]["feature"].tolist()
    if len(kept) < min(min_features, len(cols)) and not report.empty:
        report = report.sort_values(["source_minus_label", "source_probe_balanced_accuracy"], ascending=[True, True]).copy()
        kept = report.head(min(min_features, len(cols)))["feature"].tolist()
        report["kept"] = report["feature"].isin(kept)
    if mode == "iterative":
        kept = iterative_source_guard(train_df, dev_df, kept, report, source_margin=source_margin, min_features=min_features)
        report["kept"] = report["feature"].isin(kept)
    return kept, report.sort_values("feature").reset_index(drop=True)


def _multifeature_probe_accuracy(train_df: pd.DataFrame, dev_df: pd.DataFrame, cols: list[str], target: str) -> float:
    if not cols or target not in train_df.columns or train_df[target].nunique() < 2 or dev_df[target].nunique() < 2:
        return np.nan
    med = train_df[cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    x_train = train_df[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_dev = dev_df[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y_train = train_df[target].astype(str) if target != "label" else train_df[target].astype(int)
    y_dev = dev_df[target].astype(str) if target != "label" else dev_df[target].astype(int)
    clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))])
    clf.fit(x_train, y_train)
    return float(clf.score(x_dev, y_dev))


def iterative_source_guard(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    cols: list[str],
    report: pd.DataFrame,
    *,
    source_margin: float,
    min_features: int,
) -> list[str]:
    kept = list(cols)
    ranked = report.set_index("feature").reindex(kept).sort_values(
        ["source_minus_label", "source_probe_balanced_accuracy"], ascending=[False, False]
    )
    removal_queue = [c for c in ranked.index.tolist() if c in kept]
    target_min = min(int(min_features), len(kept))
    while len(kept) > target_min:
        label_acc = _multifeature_probe_accuracy(train_df, dev_df, kept, "label")
        source_acc = _multifeature_probe_accuracy(train_df, dev_df, kept, "source_dataset")
        if pd.isna(source_acc) or pd.isna(label_acc) or source_acc <= label_acc + source_margin:
            break
        if not removal_queue:
            break
        victim = removal_queue.pop(0)
        if victim in kept:
            kept.remove(victim)
    return kept


def leakage_audit(cols: list[str]) -> dict:
    bad = [c for c in cols if any(term in c.lower() for term in BANNED_FEATURE_TERMS)]
    return {
        "passed": not bad,
        "banned_feature_names": bad,
        "policy": "Downstream classifier may only use local-K spectral/residual/trajectory scalar features.",
        "banned_terms": BANNED_FEATURE_TERMS,
    }


def source_probe_risks(probe_rows: list[dict]) -> list[dict]:
    if not probe_rows:
        return []
    df = pd.DataFrame(probe_rows)
    risks = []
    for exp, grp in df.groupby("experiment"):
        scores = dict(zip(grp["probe"], grp["dev_accuracy"]))
        if "source" in scores and "label" in scores and float(scores["source"]) > float(scores["label"]):
            risks.append(
                {
                    "experiment": exp,
                    "risk": "source_artifact_risk",
                    "source_probe_dev_accuracy": float(scores["source"]),
                    "label_probe_dev_accuracy": float(scores["label"]),
                }
            )
    return risks


def write_transition_delta(summary: pd.DataFrame, out_path: Path) -> None:
    refs = []
    for path in [
        ROOT / "results_transition" / "qwen25_7b_targeted" / "transition_7b_vs_1_5b_comparison.csv",
        ROOT / "results_transition" / "qwen25_7b_targeted" / "transition_7b_summary.csv",
        ROOT / "results_transition" / "fullscale_1_5b_optimized" / "transition_optimized_summary.csv",
    ]:
        if path.exists():
            refs.append(pd.read_csv(path))
    if not refs or summary.empty or "status" not in summary:
        write_csv(pd.DataFrame(), out_path)
        return
    ref = pd.concat(refs, ignore_index=True)
    ref = ref[ref["experiment"].astype(str).str.contains("transition", case=False, na=False)].copy()
    if ref.empty:
        write_csv(pd.DataFrame(), out_path)
        return
    ref = ref.sort_values(["auroc", "auprc", "tpr_at_fpr_5pct"], ascending=False).drop_duplicates(["train_name", "test_set"])
    cur = summary[summary["status"].eq("ok")].copy()
    rows = []
    for _, row in cur.iterrows():
        rr = ref[(ref["train_name"].astype(str).eq(str(row["train_source"]))) & (ref["test_set"].astype(str).eq(str(row["test_set"])))]
        if rr.empty:
            continue
        base = rr.iloc[0]
        rows.append(
            {
                "train_source": row["train_source"],
                "test_set": row["test_set"],
                "feature_set": row["feature_set"],
                "text_koopman_auroc": row.get("auroc", np.nan),
                "transition_auroc": base.get("auroc", np.nan),
                "delta_auroc": row.get("auroc", np.nan) - base.get("auroc", np.nan),
                "text_koopman_auprc": row.get("auprc", np.nan),
                "transition_auprc": base.get("auprc", np.nan),
                "delta_auprc": row.get("auprc", np.nan) - base.get("auprc", np.nan),
                "text_koopman_tpr_at_fpr_5pct": row.get("tpr_at_fpr_5pct", np.nan),
                "transition_tpr_at_fpr_5pct": base.get("tpr_at_fpr_5pct", np.nan),
                "delta_tpr_at_fpr_5pct": row.get("tpr_at_fpr_5pct", np.nan) - base.get("tpr_at_fpr_5pct", np.nan),
                "transition_reference_experiment": base.get("experiment", ""),
            }
        )
    write_csv(pd.DataFrame(rows), out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--hidden_dir", default="features_hidden_states")
    parser.add_argument("--feature_dir", default="features_text_koopman")
    parser.add_argument("--checkpoint_dir", default="checkpoints_text_koopman")
    parser.add_argument("--output_dir", default="results_text_koopman")
    parser.add_argument("--train_sources", nargs="+", default=["m4", "leave_out_ghostbuster", "combined_strict"])
    parser.add_argument("--test_sets", nargs="+", default=TEST_SETS)
    parser.add_argument("--models", nargs="+", default=["qwen25_1_5b"])
    parser.add_argument("--projection_dims", nargs="+", type=int, default=[128])
    parser.add_argument("--projector_type", choices=["random", "pca", "linear"], default="random")
    parser.add_argument("--observable_dims", nargs="+", type=int, default=[512])
    parser.add_argument("--dmd_ranks", nargs="+", type=int, default=[32])
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_batch_size", type=int, default=2)
    parser.add_argument("--max_train_sequences", type=int, default=None)
    parser.add_argument("--max_dev_sequences", type=int, default=None)
    parser.add_argument("--max_effective_epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_hidden_cache", action="store_true")
    parser.add_argument("--force_hidden_cache", action="store_true")
    parser.add_argument("--run_lifting_train", action="store_true")
    parser.add_argument("--run_feature_extract", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--run_plots", action="store_true")
    parser.add_argument("--source_guard", action="store_true", help="Filter spectral scalar features with public train/dev source probes before downstream classification.")
    parser.add_argument("--source_guard_max_source_acc", type=float, default=0.75)
    parser.add_argument("--source_guard_source_margin", type=float, default=0.02)
    parser.add_argument("--source_guard_min_features", type=int, default=12)
    parser.add_argument("--source_guard_mode", choices=["univariate", "iterative"], default="univariate")
    parser.add_argument("--classifier_mode", choices=["classical", "fast_logistic"], default="classical")
    args = parser.parse_args()

    data = build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    needed = dataset_names_needed(args.train_sources, args.test_sets)
    result_dir = ROOT / args.output_dir
    ckpt_root = ROOT / args.checkpoint_dir
    feature_root = ROOT / args.feature_dir
    plot_dir = result_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dry_info = {
        "created_at": now(),
        "needed_datasets": {k: len(data[k]) for k in needed},
        "train_sources": args.train_sources,
        "test_sets": args.test_sets,
        "models": args.models,
        "projection_dims": args.projection_dims,
        "observable_dims": args.observable_dims,
        "dmd_ranks": args.dmd_ranks,
        "strict_policy": "No pooled hidden/latent/token-id/text features enter downstream classifiers.",
    }
    if args.dry_run:
        save_json(dry_info, result_dir / "text_koopman_dry_run.json")
        print(json.dumps(dry_info, indent=2))
        return

    cache_info = ensure_hidden_caches(args, data, needed)
    summary_rows = []
    probe_rows = []
    manifest = {
        "created_at": now(),
        "run_args": vars(args),
        "hidden_cache": cache_info,
        "experiments": {},
        "errors": [],
    }
    train_features = pd.read_csv(ROOT / args.full_features)
    external_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)

    for model_name in args.models:
        for train_source in args.train_sources:
            train_meta, dev_meta = composite_train_dev(data, train_source)
            for projection_dim in args.projection_dims:
                for observable_dim in args.observable_dims:
                    if observable_dim <= projection_dim:
                        manifest["errors"].append({"train_source": train_source, "model_name": model_name, "error": "observable_dim <= projection_dim", "projection_dim": projection_dim, "observable_dim": observable_dim})
                        continue
                    for dmd_rank in args.dmd_ranks:
                        exp = experiment_name(train_source, model_name, projection_dim, args.projector_type, observable_dim, dmd_rank, args.max_length, args.max_rows_per_split)
                        ckpt = ckpt_root / exp
                        try:
                            ckpt_ok = checkpoint_compatible(
                                ckpt,
                                train_rows=len(train_meta),
                                dev_rows=len(dev_meta),
                                projection_dim=projection_dim,
                                observable_dim=observable_dim,
                                dmd_rank=dmd_rank,
                                max_length=args.max_length,
                                projector_type=args.projector_type,
                            )
                            if not ckpt_ok:
                                if not args.run_lifting_train:
                                    raise RuntimeError(f"Checkpoint is missing or incompatible with requested data scope: {ckpt}")
                                train_text_koopman_lifting(
                                    train_meta=train_meta,
                                    dev_meta=dev_meta,
                                    hidden_root=ROOT / args.hidden_dir,
                                    model_name=model_name,
                                    dataset_names=needed,
                                    output_dir=ckpt,
                                    projection_dim=projection_dim,
                                    projector_type=args.projector_type,
                                    observable_dim=observable_dim,
                                    dmd_rank=dmd_rank,
                                    max_seq_len=args.max_length,
                                    epochs=args.epochs,
                                    patience=args.patience,
                                    batch_size=args.batch_size,
                                    max_train_sequences=args.max_train_sequences,
                                    max_dev_sequences=args.max_dev_sequences,
                                    max_effective_epochs=args.max_effective_epochs,
                                    seed=args.seed,
                                    device=args.device,
                                )
                            lifting, projector, meta = load_feature_artifacts(ckpt, device="cpu")
                            if args.run_feature_extract:
                                feature_device = "cuda" if (args.device == "auto" and __import__("torch").cuda.is_available()) else args.device
                                for ds in needed:
                                    out_csv = feature_csv(feature_root, exp, ds)
                                    if feature_csv_complete(out_csv, len(data[ds]), checkpoint_created_at=meta.get("created_at")):
                                        continue
                                    extract_text_koopman_features(
                                        hidden_root=ROOT / args.hidden_dir,
                                        model_name=model_name,
                                        dataset_name=ds,
                                        lifting_model=lifting,
                                        projector=projector,
                                        output_csv=out_csv,
                                        dmd_rank=dmd_rank,
                                        checkpoint_created_at=meta.get("created_at"),
                                        allowed_ids=set(data[ds]["id"].astype(str)) if args.max_rows_per_split is not None else None,
                                        device=feature_device,
                                    )
                                write_composite_features(feature_root, exp, train_source, data)
                            manifest["experiments"][exp] = {"status": "ok", "checkpoint": str(ckpt), "metadata": meta}
                        except Exception as exc:
                            manifest["errors"].append({"experiment": exp, "error": repr(exc)})
                            continue
                        if not args.run_eval:
                            continue
                        pool = pool_features(feature_root, exp, BASE_DATASETS + [f"{train_source}_train", f"{train_source}_dev"])
                        train_base = add_features(train_meta, pool)
                        dev_base = add_features(dev_meta, pool)
                        spec_cols = text_koopman_cols(train_base)
                        source_guard_report = None
                        if args.source_guard:
                            spec_cols, source_guard_report = source_guarded_spectral_cols(
                                train_base,
                                dev_base,
                                spec_cols,
                                max_source_acc=args.source_guard_max_source_acc,
                                source_margin=args.source_guard_source_margin,
                                min_features=args.source_guard_min_features,
                                mode=args.source_guard_mode,
                            )
                            guard_path = result_dir / f"{exp}_source_guard_feature_report.csv"
                            write_csv(source_guard_report, guard_path)
                            manifest["experiments"].setdefault(exp, {}).update(
                                {
                                    "source_guard": {
                                        "enabled": True,
                                        "n_features_after_guard": len(spec_cols),
                                        "report": str(guard_path),
                                        "max_source_acc": args.source_guard_max_source_acc,
                                        "source_margin": args.source_guard_source_margin,
                                        "mode": args.source_guard_mode,
                                    }
                                }
                            )
                        audit = leakage_audit(spec_cols)
                        save_json(audit, result_dir / "leakage_audit.json")
                        if not audit["passed"]:
                            raise RuntimeError(f"Leakage audit failed: {audit['banned_feature_names']}")
                        probe_rows.extend({"experiment": exp, "train_source": train_source, **r} for r in run_probe(train_base, dev_base, spec_cols))
                        plot_spectral_pca(pd.concat([dev_base, add_features(data["all_samples"], pool)], ignore_index=True), spec_cols, plot_dir)
                        feature_sets = {"text_koopman_spectral_only": spec_cols}
                        full_train = merge_full_features(train_meta, train_features, external_features, full_cols)
                        full_dev = merge_full_features(dev_meta, train_features, external_features, full_cols)
                        train_full_text = full_train.merge(train_base[["id", *spec_cols]], on="id", how="left")
                        dev_full_text = full_dev.merge(dev_base[["id", *spec_cols]], on="id", how="left")
                        feature_sets["full_plus_text_koopman_spectral"] = full_cols + spec_cols
                        trans_train = merge_transition_features(train_base, train_source, args.models)
                        trans_dev = merge_transition_features(dev_base, train_source, args.models)
                        trans_cols = transition_cols(trans_train)
                        if trans_cols:
                            feature_sets["transition_plus_text_koopman_spectral"] = trans_cols + spec_cols
                            train_full_trans_text = merge_transition_features(train_full_text, train_source, args.models)
                            dev_full_trans_text = merge_transition_features(dev_full_text, train_source, args.models)
                            feature_sets["full_plus_transition_plus_text_koopman_spectral"] = full_cols + trans_cols + spec_cols
                        else:
                            train_full_trans_text = train_full_text
                            dev_full_trans_text = dev_full_text
                        train_map = {
                            "text_koopman_spectral_only": train_base,
                            "full_plus_text_koopman_spectral": train_full_text,
                            "transition_plus_text_koopman_spectral": trans_train,
                            "full_plus_transition_plus_text_koopman_spectral": train_full_trans_text,
                        }
                        dev_map = {
                            "text_koopman_spectral_only": dev_base,
                            "full_plus_text_koopman_spectral": dev_full_text,
                            "transition_plus_text_koopman_spectral": trans_dev,
                            "full_plus_transition_plus_text_koopman_spectral": dev_full_trans_text,
                        }
                        for fs, cols in feature_sets.items():
                            if not cols:
                                continue
                            try:
                                if args.classifier_mode == "fast_logistic":
                                    best_name, best_model, med, selection_rows = fit_logistic_fast(train_map[fs], dev_map[fs], cols)
                                    selection_df = pd.DataFrame([{"model": best_name, "selection_mode": "fast_logistic"}])
                                else:
                                    best_name, best_model, med, selection_df = fit_select_classical(train_map[fs], dev_map[fs], cols, seed=args.seed)
                                    selection_df["selection_mode"] = "classical_dev_auprc_auroc_tpr5_f1"
                                selection_path = result_dir / f"{exp}_{fs}_classifier_selection.csv"
                                write_csv(selection_df, selection_path)
                                joblib.dump({"model": best_model, "medians": med, "columns": cols, "feature_set": fs, "selection": str(selection_path)}, ckpt / f"{fs}_classifier.joblib")
                                for test_name in args.test_sets:
                                    meta = test_df_for(data, test_name)
                                    test_df = add_features(meta, pool)
                                    if fs.startswith("full_plus"):
                                        test_df = merge_full_features(meta, train_features, external_features, full_cols).merge(test_df[["id", *spec_cols]], on="id", how="left")
                                    if "transition" in fs:
                                        test_df = merge_transition_features(test_df, train_source, args.models)
                                    out = result_dir / f"{exp}_{fs}_to_{test_name}"
                                    m = eval_one(best_model, test_df, cols, med, out)
                                    write_csv(pd.DataFrame([m]), out / "detector_metrics.csv")
                                    summary_rows.append({"status": "ok", "experiment": exp, "model_name": model_name, "train_source": train_source, "feature_set": fs, "test_set": test_name, "best_model": best_name, "n_features": len(cols), **m})
                            except Exception as exc:
                                summary_rows.append({"status": "error", "experiment": exp, "model_name": model_name, "train_source": train_source, "feature_set": fs, "test_set": "all", "error": repr(exc)})

    summary = pd.DataFrame(summary_rows)
    write_csv(summary, result_dir / "text_koopman_summary.csv")
    if not summary.empty:
        write_csv(summary[summary.get("test_set", "").eq("all_samples")] if "test_set" in summary else pd.DataFrame(), result_dir / "text_koopman_all_samples_summary.csv")
        source = summary[summary["status"].eq("ok")][["train_source", "test_set", "feature_set", "auroc", "auprc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "fpr_at_tpr_95pct", "expected_calibration_error", "brier_score"]].copy() if "status" in summary else pd.DataFrame()
        write_csv(source, result_dir / "text_koopman_source_matrix.csv")
        prev = {"model_version": "previous_best_transition", "train_source": "leave_out_ghostbuster", "feature_set": "full_plus_1_5b_and_7b_transition", "auroc": 0.6951, "auprc": 0.6592, "tpr_at_fpr_5pct": 0.0933}
        best = summary[(summary["status"].eq("ok")) & (summary["test_set"].eq("all_samples"))].sort_values(["auroc", "auprc"], ascending=False).head(10).copy() if "status" in summary else pd.DataFrame()
        comp = pd.concat([pd.DataFrame([prev]), best], ignore_index=True)
        write_csv(comp, result_dir / "text_koopman_vs_previous_best.csv")
        write_transition_delta(summary, result_dir / "text_koopman_vs_transition_delta.csv")
        if args.run_plots:
            plot_transition_delta_heatmap(result_dir / "text_koopman_vs_transition_delta.csv", plot_dir)
            plot_outputs(summary, plot_dir)
    write_csv(pd.DataFrame(probe_rows), result_dir / "probe_summary.csv")
    final_audit = json.loads((result_dir / "leakage_audit.json").read_text(encoding="utf-8")) if (result_dir / "leakage_audit.json").exists() else leakage_audit([])
    final_audit["source_probe_risks"] = source_probe_risks(probe_rows)
    final_audit["source_artifact_risk"] = bool(final_audit["source_probe_risks"])
    save_json(final_audit, result_dir / "leakage_audit.json")
    manifest["completed_at"] = now()
    manifest["summary_rows"] = int(len(summary_rows))
    save_json(manifest, result_dir / "text_koopman_manifest.json")


if __name__ == "__main__":
    main()
