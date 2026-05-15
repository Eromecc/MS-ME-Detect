#!/usr/bin/env python3
"""Evaluate a saved checkpoint on an external feature matrix."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_eval import (
    detector_metrics,
    probabilities,
    save_calibration_curve,
    save_pr_curve,
    save_roc_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on external test features.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--test_feature_file", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(checkpoint_dir / "best_model.joblib")
    feature_columns = json.loads((checkpoint_dir / "feature_columns.json").read_text(encoding="utf-8"))
    medians = json.loads((checkpoint_dir / "feature_medians.json").read_text(encoding="utf-8"))

    features = pd.read_csv(args.test_feature_file)
    test_df = pd.read_csv(args.test_csv)
    merged = test_df.merge(features, on="id", how="left")
    if "label" not in merged.columns:
        if "label_x" in merged.columns:
            merged["label"] = merged["label_x"]
        elif "label_y" in merged.columns:
            merged["label"] = merged["label_y"]
    for col in ["source_dataset", "domain", "generator", "attack_type", "text"]:
        if col not in merged.columns:
            for suffix_col in [f"{col}_x", f"{col}_y"]:
                if suffix_col in merged.columns:
                    merged[col] = merged[suffix_col]
                    break
    for col in feature_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    x = merged[feature_columns].copy()
    for col in feature_columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(medians.get(col, 0.0))

    prob = probabilities(model, x)
    pred = (prob >= 0.5).astype(int)
    preds = merged.copy()
    preds["ai_probability"] = prob
    preds["prediction"] = pred
    keep_cols = [c for c in ["id", "text", "label", "ai_probability", "prediction", "source_dataset", "domain", "generator", "attack_type"] if c in preds.columns]
    preds[keep_cols].to_csv(output_dir / "predictions.csv", index=False)

    has_label = "label" in merged.columns and merged["label"].notna().any()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "test_feature_file": str(args.test_feature_file),
        "test_csv": str(args.test_csv),
        "output_dir": str(output_dir),
        "has_label": bool(has_label),
        "rows": int(len(merged)),
    }

    if has_label:
        y_true = pd.to_numeric(merged["label"], errors="coerce").dropna().astype(int)
        valid_mask = merged["label"].notna()
        y_prob = prob[valid_mask.to_numpy()]
        y_pred = pred[valid_mask.to_numpy()]
        metrics = detector_metrics(y_true, y_prob, y_pred=y_pred, threshold=0.5)
        pd.DataFrame([metrics]).to_csv(output_dir / "detector_metrics.csv", index=False)
        pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
        save_roc_curve(y_true, y_prob, output_dir).to_csv(output_dir / "roc_curve.csv", index=False)
        save_pr_curve(y_true, y_prob, output_dir).to_csv(output_dir / "pr_curve.csv", index=False)
        save_calibration_curve(y_true, y_prob, output_dir).to_csv(output_dir / "calibration_bins.csv", index=False)
        with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
            handle.write(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Human", "AI"], zero_division=0))
        manifest["metrics"] = metrics
    else:
        manifest["metrics"] = None

    (output_dir / "external_eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    print(f"Has label: {has_label}")


if __name__ == "__main__":
    main()
