"""Train ML classifiers, evaluate metrics, ablations, and feature importance."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
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


def candidate_models() -> dict[str, object]:
    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced_subsample"),
    }
    if has_package("xgboost"):
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=config.RANDOM_STATE)
        except Exception:
            pass
    return models


def probabilities(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return pred.astype(float)


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict[str, float | str]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = probabilities(model, x_test)
    out = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_test, prob) if len(set(y_test)) > 1 else np.nan
    except Exception:
        out["roc_auc"] = np.nan
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


def train_and_evaluate(input_path: str | Path, result_dir: str | Path) -> dict[str, object]:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    cols = feature_columns(df)
    if not cols:
        raise ValueError("No numeric feature columns found.")
    x_train, x_test, y_train, y_test, train_df, test_df = split_xy(df, cols)
    metrics = []
    best = None
    for name, model in candidate_models().items():
        m = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        metrics.append(m)
        if best is None or (m.get("f1", 0), m.get("roc_auc", 0)) > (best[0].get("f1", 0), best[0].get("roc_auc", 0)):
            best = (m, model, name)
    assert best is not None
    best_metrics, best_model, best_name = best
    pred = best_model.predict(x_test)
    prob = probabilities(best_model, x_test)
    write_csv(pd.DataFrame(metrics), result_dir / "metrics.csv")
    with open(result_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(classification_report(y_test, pred, target_names=["Human", "AI"], zero_division=0))
    preds = test_df[["id", "text", "label", "type", "source", "topic"]].copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    write_csv(preds, result_dir / "predictions.csv")
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
    return {"best_model": best_name, "metrics": best_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.FEATURE_DIR / "all_features.csv"))
    parser.add_argument("--result_dir", default=str(config.RESULT_DIR))
    args = parser.parse_args()
    config.ensure_dirs()
    info = train_and_evaluate(args.input, args.result_dir)
    print(info)


if __name__ == "__main__":
    main()
