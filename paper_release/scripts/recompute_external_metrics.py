#!/usr/bin/env python3
"""Recompute external metrics from anonymous hash-only prediction scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "sample_hash",
    "label",
    "score_final",
    "score_no_segment",
    "score_qwen14_segment",
    "pred_final_0_5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute paper external metrics from anonymous prediction scores."
    )
    parser.add_argument("--input", required=True, help="Hash-only prediction CSV.")
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of ECE bins.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Input file has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise SystemExit(
                "Input file is missing required columns: " + ", ".join(missing)
            )

        labels: list[int] = []
        scores: list[float] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                score = float(row["score_final"])
            except ValueError as exc:
                raise SystemExit(f"Invalid numeric value at line {line_number}: {exc}")
            if label not in (0, 1):
                raise SystemExit(f"Invalid label at line {line_number}: {label}")
            if not math.isfinite(score):
                raise SystemExit(f"Non-finite score at line {line_number}: {score}")
            labels.append(label)
            scores.append(score)

    if not labels:
        raise SystemExit(f"Input file has no prediction rows: {path}")
    if len(set(labels)) < 2:
        raise SystemExit("Both positive and negative labels are required.")
    return labels, scores


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_auc(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(labels)
    tp = 0
    total = 0
    precision_sum = 0.0
    for score, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True):
        total += 1
        if label == 1:
            tp += 1
            precision_sum += tp / total
    return precision_sum / n_pos


def tpr_at_fpr(labels: list[int], scores: list[float], max_fpr: float = 0.05) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp = 0
    fp = 0
    best = 0.0
    for score, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True):
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        if fpr <= max_fpr:
            best = max(best, tp / n_pos)
    return best


def brier_score(labels: list[int], scores: list[float]) -> float:
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def expected_calibration_error(labels: list[int], scores: list[float], bins: int) -> float:
    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            (label, score)
            for label, score in zip(labels, scores)
            if (score >= lower and (score < upper or (bin_index == bins - 1 and score <= upper)))
        ]
        if not members:
            continue
        avg_score = sum(score for _, score in members) / len(members)
        frac_pos = sum(label for label, _ in members) / len(members)
        ece += len(members) / total * abs(frac_pos - avg_score)
    return ece


def main() -> int:
    args = parse_args()
    labels, scores = read_rows(Path(args.input))
    metrics = binary_metrics(labels, scores, args.threshold)
    metrics.update(
        {
            "roc_auc": roc_auc(labels, scores),
            "auprc": average_precision(labels, scores),
            "tpr_at_fpr_le_5_percent": tpr_at_fpr(labels, scores, 0.05),
            "brier": brier_score(labels, scores),
            "ece": expected_calibration_error(labels, scores, args.ece_bins),
            "n": len(labels),
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
        }
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
