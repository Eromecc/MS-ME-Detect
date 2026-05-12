"""Multi-scale probability response features.

This module derives explainable response features from existing Qwen2.5 Base
probability feature files. It models how PPL and token-level loss statistics
change as model scale increases from 1.5B to 7B, 14B, and optionally 32B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-8

SCALES = [
    {
        "label": "1_5b",
        "index": 0.0,
        "file": "probability_qwen25_1_5b.csv",
        "prefixes": ["qwen25_1_5b", "qwen2_5_1_5b"],
    },
    {
        "label": "7b",
        "index": 1.0,
        "file": "probability_qwen25_7b.csv",
        "prefixes": ["qwen25_7b", "qwen2_5_7b"],
    },
    {
        "label": "14b",
        "index": 2.0,
        "file": "probability_qwen25_14b.csv",
        "prefixes": ["qwen25_14b", "qwen2_5_14b"],
    },
    {
        "label": "32b",
        "index": 3.0,
        "file": "probability_qwen25_32b.csv",
        "prefixes": ["qwen25_32b", "qwen2_5_32b"],
    },
]

METRICS = ["ppl", "loss_mean", "loss_std"]


def safe_divide(a, b):
    """Elementwise safe division with NaN preservation."""
    return pd.to_numeric(a, errors="coerce") / (pd.to_numeric(b, errors="coerce") + EPSILON)


def safe_slope(v1, v2, scale_gap):
    """Elementwise slope between two model scales."""
    return (pd.to_numeric(v2, errors="coerce") - pd.to_numeric(v1, errors="coerce")) / (float(scale_gap) + EPSILON)


def response_area(values, scale_indices):
    """Trapezoidal response area over available non-NaN scale points."""
    vals = np.asarray(values, dtype=float)
    idx = np.asarray(scale_indices, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(idx)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(vals[mask], idx[mask]))


def response_curvature(values):
    """Mean second-order finite difference over available scale points."""
    vals = np.asarray(values, dtype=float)
    indices = np.asarray([s["index"] for s in SCALES[: len(vals)]], dtype=float)
    mask = np.isfinite(vals) & np.isfinite(indices)
    if mask.sum() < 3:
        return np.nan
    x = indices[mask]
    y = vals[mask]
    first = np.gradient(y, x)
    second = np.gradient(first, x)
    return float(np.nanmean(second))


def warn(message: str) -> None:
    print(f"Warning: {message}")


def find_metric_column(columns: list[str], prefixes: list[str], metric: str) -> str | None:
    """Find a metric column for a model prefix, tolerating minor naming shifts."""
    candidates = []
    for prefix in prefixes:
        candidates.append(f"{prefix}_{metric}")
    for candidate in candidates:
        if candidate in columns:
            return candidate
    suffix = f"_{metric}"
    matches = [c for c in columns if c.endswith(suffix) and any(c.startswith(p) for p in prefixes)]
    return matches[0] if matches else None


def load_probability_features(feature_dir: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """Load and merge available probability files by id."""
    feature_dir = Path(feature_dir)
    merged: pd.DataFrame | None = None
    available = []
    for scale in SCALES:
        path = feature_dir / scale["file"]
        if not path.exists():
            warn(f"Missing probability file: {path}")
            continue
        frame = pd.read_csv(path)
        if "id" not in frame.columns:
            warn(f"Skipping {path}; no id column found.")
            continue
        frame["id"] = frame["id"].astype(str)
        columns = frame.columns.tolist()
        metric_cols = {}
        for metric in METRICS:
            col = find_metric_column(columns, scale["prefixes"], metric)
            if col is None:
                warn(f"{path.name} has no {metric} column for prefixes {scale['prefixes']}.")
            metric_cols[metric] = col
        keep_cols = ["id"] + [c for c in metric_cols.values() if c is not None]
        renamed = frame[keep_cols].copy()
        for metric, col in metric_cols.items():
            if col is not None:
                renamed = renamed.rename(columns={col: f"{scale['label']}__{metric}"})
        merged = renamed if merged is None else merged.merge(renamed, on="id", how="outer")
        available.append(scale)
    if merged is None:
        warn("No probability feature files were available; writing an empty id-only feature file.")
        return pd.DataFrame({"id": pd.Series(dtype=str)}), []
    return merged, available


def add_pair_features(out: pd.DataFrame, merged: pd.DataFrame, left: dict, right: dict) -> None:
    gap = right["index"] - left["index"]
    l_label = left["label"]
    r_label = right["label"]
    for metric in METRICS:
        l_col = f"{l_label}__{metric}"
        r_col = f"{r_label}__{metric}"
        if l_col not in merged.columns or r_col not in merged.columns:
            continue
        out[f"scale_{metric}_slope_{l_label}_to_{r_label}"] = safe_slope(merged[l_col], merged[r_col], gap)
        out[f"scale_{metric}_ratio_{l_label}_{r_label}"] = safe_divide(merged[l_col], merged[r_col])


def add_gap_features(out: pd.DataFrame, merged: pd.DataFrame, small: dict, large: dict) -> None:
    s_label = small["label"]
    l_label = large["label"]
    for metric in METRICS:
        s_col = f"{s_label}__{metric}"
        l_col = f"{l_label}__{metric}"
        if s_col in merged.columns and l_col in merged.columns:
            out[f"scale_{metric}_gap_{s_label}_{l_label}"] = pd.to_numeric(merged[s_col], errors="coerce") - pd.to_numeric(merged[l_col], errors="coerce")


def add_response_shape_features(out: pd.DataFrame, merged: pd.DataFrame, available: list[dict]) -> None:
    scale_indices = [s["index"] for s in available]
    for metric in METRICS:
        cols = [f"{s['label']}__{metric}" for s in available]
        existing = [c for c in cols if c in merged.columns]
        if len(existing) < 2:
            out[f"scale_{metric}_response_area"] = np.nan
            out[f"scale_{metric}_response_curvature"] = np.nan
            continue
        idx = [scale_indices[cols.index(c)] for c in existing]
        values = merged[existing].apply(pd.to_numeric, errors="coerce")
        out[f"scale_{metric}_response_area"] = values.apply(lambda row: response_area(row.to_numpy(), idx), axis=1)
        out[f"scale_{metric}_response_curvature"] = values.apply(lambda row: response_curvature(row.to_numpy()), axis=1)


def add_saturation_features(out: pd.DataFrame, merged: pd.DataFrame) -> None:
    for metric in METRICS:
        c14 = f"14b__{metric}"
        c32 = f"32b__{metric}"
        if c14 in merged.columns and c32 in merged.columns:
            out[f"scale_{metric}_saturation_14b_32b"] = (
                pd.to_numeric(merged[c14], errors="coerce") - pd.to_numeric(merged[c32], errors="coerce")
            ).abs()


def build_scale_response_features(feature_dir: str | Path, output: str | Path) -> pd.DataFrame:
    merged, available = load_probability_features(feature_dir)
    out = pd.DataFrame({"id": merged["id"].astype(str)}) if "id" in merged.columns else pd.DataFrame({"id": pd.Series(dtype=str)})

    by_label = {s["label"]: s for s in available}
    for left_label, right_label in [("1_5b", "7b"), ("7b", "14b"), ("14b", "32b")]:
        if left_label in by_label and right_label in by_label:
            add_pair_features(out, merged, by_label[left_label], by_label[right_label])

    if "1_5b" in by_label and "14b" in by_label:
        add_gap_features(out, merged, by_label["1_5b"], by_label["14b"])
    if "1_5b" in by_label and "32b" in by_label:
        add_gap_features(out, merged, by_label["1_5b"], by_label["32b"])

    add_response_shape_features(out, merged, available)
    if "14b" in by_label and "32b" in by_label:
        add_saturation_features(out, merged)

    for col in out.columns:
        if col != "id":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved scale response features: {output} ({len(out)} rows, {len(out.columns) - 1} features)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-scale probability response features.")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--output", default="features/scale_response_features.csv")
    args = parser.parse_args()
    build_scale_response_features(args.feature_dir, args.output)


if __name__ == "__main__":
    main()
