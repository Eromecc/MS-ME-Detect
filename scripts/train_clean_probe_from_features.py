#!/usr/bin/env python3
"""Train shallow probe heads from one model-feature family with strict split alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {
    "id",
    "text",
    "label",
    "source_dataset",
    "domain",
    "generator",
    "source",
    "split",
}


def load_split(meta_path: Path, feature_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    feats = pd.read_csv(feature_path)
    if "id" not in meta.columns or "id" not in feats.columns:
        raise ValueError(f"missing id column: {meta_path} / {feature_path}")
    duplicate_meta_cols = [col for col in feats.columns if col != "id" and col in meta.columns]
    if duplicate_meta_cols:
        feats = feats.drop(columns=duplicate_meta_cols)
    df = meta.merge(feats, on="id", how="left", validate="one_to_one")
    if len(df) != len(meta) or len(df) != len(feats):
        raise ValueError(f"row mismatch after merge: {meta_path} / {feature_path}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, max_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if np.any(ok) else 0.0


def metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_true, score))
    auprc = float(average_precision_score(y_true, score))
    low = tpr_at_fpr(y_true, score)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_fpr5": low,
        "f1": f1,
        "composite": float(0.45 * auroc + 0.35 * auprc + 0.20 * low),
    }


def score_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return expit(np.asarray(model.decision_function(x), dtype=float))


def write_predictions(path: Path, df: pd.DataFrame, score: np.ndarray, name: str) -> None:
    out = df[[c for c in ["id", "text", "label", "source_dataset", "domain", "generator", "source", "split"] if c in df.columns]].copy()
    out["score"] = np.clip(score, 0.0, 1.0)
    out["prediction"] = (out["score"] >= 0.5).astype(int)
    out["model"] = name
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--feature_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--train_csv", default="data/reproduction_datasets/fakespot_like_train.csv")
    parser.add_argument("--val_csv", default="data/reproduction_datasets/fakespot_like_val.csv")
    parser.add_argument("--all_csv", default="data/test/all_samples_prepared.csv")
    args = parser.parse_args()

    root = Path(args.feature_root) / "fakespot_like"
    train = load_split(Path(args.train_csv), root / "train" / args.feature_name)
    val = load_split(Path(args.val_csv), root / "val" / args.feature_name)
    all_samples = load_split(Path(args.all_csv), root / "all_samples" / args.feature_name)

    cols = feature_columns(train)
    if not cols:
        raise ValueError("no numeric feature columns found")
    x_train = train[cols].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    x_val = val[cols].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)
    x_all = all_samples[cols].to_numpy(dtype=float)
    y_all = all_samples["label"].to_numpy(dtype=int)

    models = {
        f"{args.model_prefix}__logreg_C03": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", n_jobs=16),
        ),
        f"{args.model_prefix}__logreg_C1": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=16),
        ),
        f"{args.model_prefix}__sgd_a1e4": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=3000, class_weight="balanced", random_state=7),
        ),
        f"{args.model_prefix}__sgd_a3e5": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=3e-5, max_iter=3000, class_weight="balanced", random_state=11),
        ),
        f"{args.model_prefix}__hgb_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, l2_regularization=0.1, random_state=13),
        ),
        f"{args.model_prefix}__extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", n_jobs=24, random_state=17),
        ),
    }

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    policy = "train on fakespot_like train; select on fakespot_like validation; all_samples eval only; metadata aligned from original split CSVs"
    for name, model in models.items():
        model.fit(x_train, y_train)
        val_score = score_model(model, x_val)
        all_score = score_model(model, x_all)
        write_predictions(pred_dir / f"{name}__validation.csv", val, val_score, name)
        write_predictions(pred_dir / f"{name}__all_samples.csv", all_samples, all_score, name)
        vm = metrics(y_val, val_score)
        am = metrics(y_all, all_score)
        rows.append(
            {
                "model": name,
                "n_features": len(cols),
                "val_auroc": vm["auroc"],
                "val_auprc": vm["auprc"],
                "val_tpr_at_fpr5": vm["tpr_at_fpr5"],
                "val_f1": vm["f1"],
                "val_composite": vm["composite"],
                "all_samples_auroc": am["auroc"],
                "all_samples_auprc": am["auprc"],
                "all_samples_tpr_at_fpr5": am["tpr_at_fpr5"],
                "all_samples_f1": am["f1"],
                "all_samples_composite": am["composite"],
                "leakage_policy": policy,
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("val_composite", ascending=False)
    leaderboard.to_csv(out_dir / f"{args.model_prefix}_leaderboard.csv", index=False)
    leaderboard.head(1).to_csv(out_dir / f"{args.model_prefix}_selected.csv", index=False)
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
