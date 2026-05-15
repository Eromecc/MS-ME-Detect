"""Train ML classifiers, evaluate metrics, ablations, and feature importance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss as sk_log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import config
    from .utils import has_package, save_json, write_csv
except ImportError:
    import config
    from utils import has_package, save_json, write_csv


def feature_columns(df: pd.DataFrame) -> list[str]:
    metadata = set(config.METADATA_COLUMNS)
    return [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]


def split_xy(df: pd.DataFrame, cols: list[str]):
    y = df["label"].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    return train_test_split(df[cols], y, df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify)


def class_balance_info(y_train) -> dict[str, float | int | str]:
    y = pd.Series(y_train).astype(int)
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    positive_rate = float(n_positive / len(y)) if len(y) else 0.0
    scale_pos_weight = float(n_negative / max(n_positive, 1))
    return {
        "class_balance_strategy": "class_weight=balanced; xgboost.scale_pos_weight=n_negative/n_positive",
        "n_positive_train": n_positive,
        "n_negative_train": n_negative,
        "positive_rate_train": positive_rate,
        "scale_pos_weight": scale_pos_weight,
    }


def candidate_models(y_train=None) -> dict[str, object]:
    balance = class_balance_info(y_train if y_train is not None else [0, 1])
    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced"),
    }
    if has_package("xgboost"):
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                scale_pos_weight=balance["scale_pos_weight"],
            )
        except Exception:
            pass
    return models


def probabilities(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return pred.astype(float)


def safe_roc_auc(y_true, y_prob) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_auprc(y_true, y_prob) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return np.nan


def tpr_at_fpr(y_true, y_prob, target_fpr) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        return np.nan
    valid = tpr[fpr <= target_fpr + 1e-12]
    return float(np.max(valid)) if valid.size else np.nan


def fpr_at_tpr(y_true, y_prob, target_tpr) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        return np.nan
    valid = fpr[tpr >= target_tpr - 1e-12]
    return float(np.min(valid)) if valid.size else np.nan


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    _, ece = calibration_bins(y_true, y_prob, n_bins=n_bins)
    return ece


def calibration_bins(y_true, y_prob, n_bins=10) -> tuple[pd.DataFrame, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    total = max(len(y_prob_arr), 1)
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_arr >= left) & (y_prob_arr <= right)
        else:
            mask = (y_prob_arr >= left) & (y_prob_arr < right)
        count = int(mask.sum())
        if count:
            mean_prob = float(np.mean(y_prob_arr[mask]))
            observed_rate = float(np.mean(y_true_arr[mask]))
            gap = abs(mean_prob - observed_rate)
            ece += (count / total) * gap
        else:
            mean_prob = np.nan
            observed_rate = np.nan
            gap = np.nan
        rows.append(
            {
                "bin_index": idx,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_predicted_probability": mean_prob,
                "observed_positive_rate": observed_rate,
                "abs_calibration_gap": gap,
            }
        )
    return pd.DataFrame(rows), float(ece)


def detector_metrics(y_true, y_prob, y_pred=None, threshold=0.5) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred_arr = (y_prob_arr >= threshold).astype(int) if y_pred is None else np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    calibration_df, ece = calibration_bins(y_true_arr, y_prob_arr, n_bins=10)
    try:
        ll = float(sk_log_loss(y_true_arr, y_prob_arr, labels=[0, 1]))
    except Exception:
        ll = np.nan
    metrics = {
        "positive_class_rate": float(np.mean(y_true_arr == 1)),
        "mean_ai_probability_human": float(np.mean(y_prob_arr[y_true_arr == 0])) if np.any(y_true_arr == 0) else np.nan,
        "mean_ai_probability_ai": float(np.mean(y_prob_arr[y_true_arr == 1])) if np.any(y_true_arr == 1) else np.nan,
        "auroc": safe_roc_auc(y_true_arr, y_prob_arr),
        "auprc": safe_auprc(y_true_arr, y_prob_arr),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "tpr": float(recall),
        "specificity": float(specificity),
        "tnr": float(specificity),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)) if len(y_true_arr) else np.nan,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true_arr, y_prob_arr, 0.01),
        "tpr_at_fpr_5pct": tpr_at_fpr(y_true_arr, y_prob_arr, 0.05),
        "tpr_at_fpr_10pct": tpr_at_fpr(y_true_arr, y_prob_arr, 0.10),
        "fpr_at_tpr_90pct": fpr_at_tpr(y_true_arr, y_prob_arr, 0.90),
        "fpr_at_tpr_95pct": fpr_at_tpr(y_true_arr, y_prob_arr, 0.95),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "expected_calibration_error": float(ece),
        "ECE": float(ece),
        "log_loss": ll,
    }
    return metrics


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict[str, float | str]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = probabilities(model, x_test)
    out = {"model": name}
    out.update(detector_metrics(y_test, prob, y_pred=pred, threshold=0.5))
    out["roc_auc"] = out["auroc"]
    return out


def get_importance(model, cols: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        vals = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        vals = np.abs(estimator.coef_[0])
    else:
        vals = np.zeros(len(cols))
    return pd.DataFrame({"feature": cols, "importance": vals}).sort_values("importance", ascending=False)


def ablation_groups(cols: list[str]) -> dict[str, list[str]]:
    probability_raw = [c for c in cols if c.startswith(("qwen25_1_5b_", "qwen25_7b_", "qwen25_14b_", "qwen25_32b_"))]
    scale_response = [c for c in cols if c.startswith("scale_")]
    burstiness = [c for c in cols if c.startswith("burst_")]
    return {
        "burstiness_only": burstiness,
        "probability_1_5b_only": [c for c in cols if c.startswith("qwen25_1_5b_")],
        "probability_7b_only": [c for c in cols if c.startswith("qwen25_7b_")],
        "probability_14b_only": [c for c in cols if c.startswith("qwen25_14b_")],
        "multi_scale_probability": probability_raw,
        "scale_response_only": scale_response,
        "probability_raw_only": probability_raw,
        "probability_raw_scale_response": probability_raw + scale_response,
        "burstiness_scale_response": burstiness + scale_response,
        "burstiness_probability_raw_scale_response": burstiness + probability_raw + scale_response,
        "binoculars_style_only": [c for c in cols if c.startswith("bino_")],
        "structure_only": [c for c in cols if c.startswith("struct_")],
        "burstiness_structure": [c for c in cols if c.startswith(("burst_", "struct_"))],
        "burstiness_multi_scale_probability": burstiness + probability_raw,
        "burstiness_probability_binoculars": [c for c in cols if c.startswith(("burst_", "qwen25_", "bino_"))],
        "all_features": cols,
    }


def save_plots(y_test, pred, importance: pd.DataFrame, result_dir: Path) -> None:
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(result_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    top = importance.head(25).iloc[::-1]
    plt.figure(figsize=(8, max(4, len(top) * 0.25)))
    plt.barh(top["feature"], top["importance"])
    plt.tight_layout()
    plt.savefig(result_dir / "feature_importance.png", dpi=160)
    plt.close()


def save_roc_curve(y_true, y_prob, result_dir: Path) -> pd.DataFrame:
    if pd.Series(y_true).nunique() < 2:
        df = pd.DataFrame({"fpr": [], "tpr": [], "threshold": []})
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(result_dir / "roc_curve.png", dpi=160)
        plt.close()
        return df
    fpr, tpr, threshold = roc_curve(y_true, y_prob)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold})
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(result_dir / "roc_curve.png", dpi=160)
    plt.close()
    return df


def save_pr_curve(y_true, y_prob, result_dir: Path) -> pd.DataFrame:
    if pd.Series(y_true).nunique() < 2:
        df = pd.DataFrame({"recall": [], "precision": [], "threshold": []})
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "PR unavailable", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(result_dir / "pr_curve.png", dpi=160)
        plt.close()
        return df
    precision, recall, threshold = precision_recall_curve(y_true, y_prob)
    threshold = np.append(threshold, np.nan)
    df = pd.DataFrame({"recall": recall, "precision": precision, "threshold": threshold})
    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(result_dir / "pr_curve.png", dpi=160)
    plt.close()
    return df


def save_calibration_curve(y_true, y_prob, result_dir: Path) -> pd.DataFrame:
    bins_df, _ = calibration_bins(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(4, 3))
    valid = bins_df["count"] > 0
    if valid.any():
        plt.plot(
            bins_df.loc[valid, "mean_predicted_probability"],
            bins_df.loc[valid, "observed_positive_rate"],
            marker="o",
        )
    else:
        plt.text(0.5, 0.5, "Calibration unavailable", ha="center", va="center")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.tight_layout()
    plt.savefig(result_dir / "calibration_curve.png", dpi=160)
    plt.close()
    return bins_df


def model_ranking_tuple(metrics: dict[str, float | str]) -> tuple[float, float, float]:
    def score(name: str) -> float:
        value = metrics.get(name, np.nan)
        return float(value) if pd.notna(value) else float("-inf")

    return (score("auprc"), score("auroc"), score("f1"))


def train_and_evaluate(
    input_path: str | Path,
    result_dir: str | Path,
    *,
    data_csv: str | Path | None = None,
    feature_file: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    experiment_name: str | None = None,
    save_model: bool = True,
    extra_metadata: dict | None = None,
) -> dict[str, object]:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    cols = feature_columns(df)
    if not cols:
        raise ValueError("No numeric feature columns found.")
    x_train, x_test, y_train, y_test, train_df, test_df = split_xy(df, cols)
    metrics = []
    best = None
    balance = class_balance_info(y_train)
    for name, model in candidate_models(y_train).items():
        m = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        metrics.append(m)
        if best is None or model_ranking_tuple(m) > model_ranking_tuple(best[0]):
            best = (m, model, name)
    assert best is not None
    best_metrics, best_model, best_name = best
    pred = best_model.predict(x_test)
    prob = probabilities(best_model, x_test)
    write_csv(pd.DataFrame(metrics), result_dir / "metrics.csv")
    detector_df = pd.DataFrame([{**best_metrics, "model": best_name}])
    write_csv(detector_df, result_dir / "detector_metrics.csv")
    with open(result_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(classification_report(y_test, pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
    meta_cols = [c for c in ["id", "text", "label", "type", "source", "topic"] if c in test_df.columns]
    preds = test_df[meta_cols].copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    write_csv(preds, result_dir / "predictions.csv")
    write_csv(save_roc_curve(y_test, prob, result_dir), result_dir / "roc_curve.csv")
    write_csv(save_pr_curve(y_test, prob, result_dir), result_dir / "pr_curve.csv")
    write_csv(save_calibration_curve(y_test, prob, result_dir), result_dir / "calibration_bins.csv")
    importance = get_importance(best_model, cols)
    write_csv(importance, result_dir / "feature_importance.csv")
    save_plots(y_test, pred, importance, result_dir)
    ablations = []
    for group, group_cols in ablation_groups(cols).items():
        if not group_cols:
            continue
        model = RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, class_weight="balanced_subsample")
        m = evaluate_model(group, model, x_train[group_cols], x_test[group_cols], y_train, y_test)
        m["feature_count"] = len(group_cols)
        ablations.append(m)
    write_csv(pd.DataFrame(ablations), result_dir / "ablation_results.csv")
    joblib.dump(best_model, result_dir / "best_model.pkl")
    save_json(cols, result_dir / "feature_columns.json")
    medians = df[cols].median(numeric_only=True).fillna(0.0).to_dict()
    save_json(medians, result_dir / "feature_medians.json")
    if save_model and checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, checkpoint_path / "best_model.joblib")
        save_json(cols, checkpoint_path / "feature_columns.json")
        save_json(medians, checkpoint_path / "feature_medians.json")
        metadata = {
            "experiment_name": experiment_name or Path(input_path).stem,
            "train_csv": str(data_csv or ""),
            "feature_file": str(feature_file or input_path),
            "best_model_name": best_name,
            "selection_metric": "auprc>auroc>f1",
            "auprc": best_metrics.get("auprc"),
            "auroc": best_metrics.get("auroc"),
            "f1": best_metrics.get("f1"),
            "n_train": int(len(x_train)),
            "n_test_internal": int(len(x_test)),
            "n_features": int(len(cols)),
            "label_mapping": {"0": "Human", "1": "AI-generated"},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata.update(balance)
        if extra_metadata:
            metadata.update(extra_metadata)
        save_json(metadata, checkpoint_path / "train_metadata.json")
    return {"best_model": best_name, "metrics": best_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.FEATURE_DIR / "all_features.csv"))
    parser.add_argument("--result_dir", default=str(config.RESULT_DIR))
    parser.add_argument("--data_csv", default=None)
    parser.add_argument("--feature_file", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()
    config.ensure_dirs()
    feature_input = args.feature_file or args.input
    result_dir = args.results_dir or args.result_dir
    info = train_and_evaluate(
        feature_input,
        result_dir,
        data_csv=args.data_csv,
        feature_file=feature_input,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        save_model=args.save_model,
    )
    print(info)


if __name__ == "__main__":
    main()
