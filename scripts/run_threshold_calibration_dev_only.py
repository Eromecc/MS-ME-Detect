#!/usr/bin/env python3
"""Select thresholds and fit calibration on public dev predictions only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_eval import calibration_bins, detector_metrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "ai_probability" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{path} must contain label and ai_probability columns.")
        df["prediction_file"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    out["ai_probability"] = pd.to_numeric(out["ai_probability"], errors="coerce").clip(0.0, 1.0)
    out = out[out["ai_probability"].notna()].copy()
    return out


def threshold_for_target_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    negatives = y_prob[y_true == 0]
    if negatives.size == 0:
        return 1.0
    # Choose the lowest threshold whose empirical false-positive rate is at or below target.
    candidates = np.unique(np.r_[0.0, y_prob, 1.0])
    best = 1.0
    for threshold in np.sort(candidates):
        fpr = float(np.mean(negatives >= threshold))
        if fpr <= target_fpr:
            best = float(threshold)
            break
    return best


def threshold_for_target_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> float | None:
    candidates = np.unique(np.r_[0.0, y_prob, 1.0])
    best_threshold = None
    best_recall = -1.0
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / max(int((y_true == 1).sum()), 1)
        if precision >= target_precision and recall > best_recall:
            best_threshold = float(threshold)
            best_recall = float(recall)
    return best_threshold


def select_thresholds(dev: pd.DataFrame) -> pd.DataFrame:
    y = dev["label"].to_numpy(dtype=int)
    p = dev["ai_probability"].to_numpy(dtype=float)
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best_f1_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        metrics = detector_metrics(y, p, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_f1_threshold = float(threshold)
    rows = [
        {"threshold_name": "default_0_5", "threshold": 0.5, "selection_rule": "fixed_default"},
        {"threshold_name": "dev_best_f1", "threshold": best_f1_threshold, "selection_rule": "max_f1_on_public_dev"},
        {"threshold_name": "dev_target_fpr_1pct", "threshold": threshold_for_target_fpr(y, p, 0.01), "selection_rule": "public_dev_fpr_le_0.01"},
        {"threshold_name": "dev_target_fpr_5pct", "threshold": threshold_for_target_fpr(y, p, 0.05), "selection_rule": "public_dev_fpr_le_0.05"},
    ]
    precision_threshold = threshold_for_target_precision(y, p, 0.90)
    if precision_threshold is not None:
        rows.append({"threshold_name": "dev_target_precision_90pct", "threshold": precision_threshold, "selection_rule": "public_dev_precision_ge_0.90_max_recall"})
    out = []
    for row in rows:
        metrics = detector_metrics(y, p, threshold=row["threshold"])
        out.append({**row, **{f"dev_{k}": v for k, v in metrics.items()}})
    return pd.DataFrame(out)


def apply_thresholds(thresholds: pd.DataFrame, all_samples: pd.DataFrame) -> pd.DataFrame:
    y = all_samples["label"].to_numpy(dtype=int)
    p = all_samples["ai_probability"].to_numpy(dtype=float)
    rows = []
    for _, row in thresholds.iterrows():
        metrics = detector_metrics(y, p, threshold=float(row["threshold"]))
        rows.append(
            {
                "threshold_name": row["threshold_name"],
                "threshold": float(row["threshold"]),
                "selection_rule": row["selection_rule"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def logit_transform(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def calibration_transfer(dev: pd.DataFrame, all_samples: pd.DataFrame, min_isotonic: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_dev = dev["label"].to_numpy(dtype=int)
    p_dev = dev["ai_probability"].to_numpy(dtype=float)
    y_all = all_samples["label"].to_numpy(dtype=int)
    p_all = all_samples["ai_probability"].to_numpy(dtype=float)
    rows_dev = []
    rows_all = []

    calibrators: list[tuple[str, np.ndarray, np.ndarray]] = [("uncalibrated", p_dev, p_all)]

    platt = LogisticRegression(max_iter=1000)
    platt.fit(logit_transform(p_dev), y_dev)
    calibrators.append(("platt_sigmoid", platt.predict_proba(logit_transform(p_dev))[:, 1], platt.predict_proba(logit_transform(p_all))[:, 1]))

    if len(dev) >= min_isotonic and np.unique(y_dev).size == 2:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_dev, y_dev)
        calibrators.append(("isotonic", iso.predict(p_dev), iso.predict(p_all)))

    for name, dev_prob, all_prob in calibrators:
        dev_metrics = detector_metrics(y_dev, np.asarray(dev_prob, dtype=float), threshold=0.5)
        all_metrics = detector_metrics(y_all, np.asarray(all_prob, dtype=float), threshold=0.5)
        rows_dev.append({"calibrator": name, **dev_metrics})
        rows_all.append({"calibrator": name, **all_metrics})
    return pd.DataFrame(rows_dev), pd.DataFrame(rows_all)


def transfer_report(threshold_metrics: pd.DataFrame, cal_all: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Dev-Only Threshold and Calibration Transfer",
        "",
        "Thresholds and calibrators in this report were selected or fit on public dev predictions only. all_samples labels were used only for final transfer evaluation.",
        "",
        "## Low-FPR Transfer",
        "",
    ]
    for name in ["dev_target_fpr_1pct", "dev_target_fpr_5pct"]:
        row = threshold_metrics[threshold_metrics["threshold_name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"- {name}: all_samples FPR={r.get('fpr', np.nan):.6f}, TPR={r.get('tpr', np.nan):.6f}, F1={r.get('f1', np.nan):.6f}")
    lines.extend(["", "## Calibration", ""])
    for _, row in cal_all.iterrows():
        lines.append(f"- {row['calibrator']}: all_samples ECE={row.get('ECE', np.nan):.6f}, Brier={row.get('brier_score', np.nan):.6f}, log_loss={row.get('log_loss', np.nan):.6f}")
    (output_dir / "threshold_calibration_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev_predictions", nargs="+", required=True, help="Public dev prediction CSVs. Never pass all_samples here.")
    parser.add_argument("--all_samples_predictions", required=True, help="Held-out all_samples prediction CSV for transfer evaluation only.")
    parser.add_argument("--output_dir", default="results_all_samples_scoreboard/threshold_calibration")
    parser.add_argument("--min_isotonic_samples", type=int, default=200)
    args = parser.parse_args()

    dev_paths = [Path(p) for p in args.dev_predictions]
    all_path = Path(args.all_samples_predictions)
    if any("all_samples" in str(p) for p in dev_paths):
        raise ValueError("--dev_predictions must not contain all_samples files.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = load_predictions(dev_paths)
    all_samples = load_predictions([all_path])
    thresholds = select_thresholds(dev)
    threshold_transfer = apply_thresholds(thresholds, all_samples)
    cal_dev, cal_all = calibration_transfer(dev, all_samples, args.min_isotonic_samples)

    thresholds.to_csv(output_dir / "threshold_selection_dev.csv", index=False)
    cal_dev.to_csv(output_dir / "calibration_dev_report.csv", index=False)
    threshold_transfer.to_csv(output_dir / "all_samples_threshold_transfer_metrics.csv", index=False)
    cal_all.to_csv(output_dir / "all_samples_calibrated_metrics.csv", index=False)
    bins, _ = calibration_bins(all_samples["label"].to_numpy(dtype=int), all_samples["ai_probability"].to_numpy(dtype=float), n_bins=10)
    bins.to_csv(output_dir / "all_samples_uncalibrated_calibration_bins.csv", index=False)
    transfer_report(threshold_transfer, cal_all, output_dir)
    write_json(
        output_dir / "threshold_calibration_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "leakage_policy": "thresholds_and_calibrators_fit_on_public_dev_only_all_samples_transfer_eval_only",
            "dev_predictions": [str(p) for p in dev_paths],
            "all_samples_predictions": str(all_path),
            "n_dev": int(len(dev)),
            "n_all_samples": int(len(all_samples)),
        },
    )
    print(f"Wrote dev-only threshold/calibration transfer outputs to {output_dir}")


if __name__ == "__main__":
    main()
