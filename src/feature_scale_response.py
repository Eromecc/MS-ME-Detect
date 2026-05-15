"""Build scale-response features from existing multi-scale probability CSVs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .utils import save_json
except ImportError:
    from utils import save_json

EPSILON = 1e-8

SCALES = [
    {"label": "1_5b", "index": 0.0, "file": "probability_qwen25_1_5b.csv", "prefixes": ["qwen25_1_5b", "qwen2_5_1_5b"]},
    {"label": "7b", "index": 1.0, "file": "probability_qwen25_7b.csv", "prefixes": ["qwen25_7b", "qwen2_5_7b"]},
    {"label": "14b", "index": 2.0, "file": "probability_qwen25_14b.csv", "prefixes": ["qwen25_14b", "qwen2_5_14b"]},
    {"label": "32b", "index": 3.0, "file": "probability_qwen25_32b.csv", "prefixes": ["qwen25_32b", "qwen2_5_32b"]},
]

REQUESTED_METRICS = [
    "ppl",
    "loss_mean",
    "loss_std",
    "loss_cv",
    "loss_range",
    "loss_skewness",
    "loss_kurtosis",
    "top_10_percent_loss_mean",
    "bottom_10_percent_loss_mean",
]
PAIRWISE_SCALE_PAIRS = [("1_5b", "7b"), ("7b", "14b"), ("14b", "32b")]


def warn(message: str) -> None:
    print(f"Warning: {message}")


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_divide(a, b):
    return to_numeric_series(a) / (to_numeric_series(b) + EPSILON)


def sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col == "id":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_cols = [c for c in df.columns if c != "id"]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def find_metric_column(columns: list[str], prefixes: list[str], metric: str) -> str | None:
    for prefix in prefixes:
        candidate = f"{prefix}_{metric}"
        if candidate in columns:
            return candidate
    suffix = f"_{metric}"
    matches = [c for c in columns if c.endswith(suffix) and any(c.startswith(prefix) for prefix in prefixes)]
    return matches[0] if matches else None


def load_probability_features(feature_dir: str | Path) -> tuple[pd.DataFrame, list[dict], list[str]]:
    feature_dir = Path(feature_dir)
    merged: pd.DataFrame | None = None
    available_scales: list[dict] = []
    metrics_found: set[str] = set()
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
        keep_cols = ["id"]
        rename_map = {}
        for metric in REQUESTED_METRICS:
            column = find_metric_column(frame.columns.tolist(), scale["prefixes"], metric)
            if column is None:
                warn(f"{path.name} is missing metric '{metric}'.")
                continue
            keep_cols.append(column)
            rename_map[column] = f"{scale['label']}__{metric}"
            metrics_found.add(metric)
        renamed = frame[keep_cols].rename(columns=rename_map).copy()
        renamed = sanitize_numeric_frame(renamed)
        merged = renamed if merged is None else merged.merge(renamed, on="id", how="outer")
        available_scales.append(scale)
    if merged is None:
        warn("No probability feature files were available; writing an empty id-only feature file.")
        return pd.DataFrame({"id": pd.Series(dtype=str)}), [], []
    return merged, available_scales, [metric for metric in REQUESTED_METRICS if metric in metrics_found]


def pairwise_directional_change(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    delta = to_numeric_series(right) - to_numeric_series(left)
    slope = delta
    ratio = safe_divide(right, left)
    abs_delta = delta.abs()
    rel_delta = safe_divide(delta, left.abs())
    return slope, ratio, abs_delta, rel_delta


def row_valid_points(row: pd.Series, scale_lookup: dict[str, dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for scale in SCALES:
        col = f"{scale['label']}__{metric}"
        if col not in row.index:
            continue
        value = row[col]
        if pd.notna(value) and np.isfinite(value):
            xs.append(scale_lookup[scale["label"]]["index"])
            ys.append(float(value))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def linear_response_stats(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    if xs.size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(xs, ys, deg=1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = np.nan if ss_tot <= EPSILON else 1.0 - (ss_res / (ss_tot + EPSILON))
    return float(slope), float(intercept), float(r2)


def response_area(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return np.nan
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(ys, xs))
    return float(np.sum((ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1]) * 0.5))


def normalized_area(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return np.nan
    span = xs[-1] - xs[0]
    if span <= EPSILON:
        return np.nan
    return response_area(xs, ys) / (span + EPSILON)


def response_curvature(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 3:
        return np.nan
    first = np.gradient(ys, xs)
    second = np.gradient(first, xs)
    return float(np.nanmean(second))


def monotonicity_stats(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    if ys.size < 2:
        return np.nan, np.nan, np.nan
    delta = np.diff(ys)
    nonzero = delta[np.abs(delta) > EPSILON]
    decrease_ratio = float(np.mean(delta < 0)) if delta.size else np.nan
    increase_ratio = float(np.mean(delta > 0)) if delta.size else np.nan
    if nonzero.size < 2:
        direction_changes = 0.0 if nonzero.size == 1 else np.nan
    else:
        direction_changes = float(np.sum(np.sign(nonzero[1:]) != np.sign(nonzero[:-1])))
    return decrease_ratio, increase_ratio, direction_changes


def early_late_saturation(row: pd.Series, metric: str) -> tuple[float, float, float, float]:
    c1 = row.get(f"1_5b__{metric}")
    c7 = row.get(f"7b__{metric}")
    c14 = row.get(f"14b__{metric}")
    c32 = row.get(f"32b__{metric}")
    early_gain = np.nan if pd.isna(c1) or pd.isna(c7) else float(c1) - float(c7)
    late_gain = np.nan if pd.isna(c14) or pd.isna(c32) else float(c14) - float(c32)
    saturation = np.nan if np.isnan(late_gain) else abs(float(late_gain))
    saturation_ratio = np.nan if np.isnan(saturation) else float(saturation / (abs(float(early_gain)) + EPSILON))
    return early_gain, late_gain, saturation, saturation_ratio


def add_pairwise_response_features(out: pd.DataFrame, merged: pd.DataFrame, metrics_used: list[str]) -> None:
    by_label = {scale["label"]: scale for scale in SCALES}
    for left_label, right_label in PAIRWISE_SCALE_PAIRS:
        left_scale = by_label[left_label]
        right_scale = by_label[right_label]
        gap = right_scale["index"] - left_scale["index"]
        for metric in metrics_used:
            left_col = f"{left_label}__{metric}"
            right_col = f"{right_label}__{metric}"
            if left_col not in merged.columns or right_col not in merged.columns:
                continue
            slope, ratio, abs_delta, rel_delta = pairwise_directional_change(merged[left_col], merged[right_col])
            out[f"scale_{metric}_slope_{left_label}_to_{right_label}"] = slope / (gap + EPSILON)
            out[f"scale_{metric}_ratio_{left_label}_to_{right_label}"] = ratio
            out[f"scale_{metric}_abs_delta_{left_label}_to_{right_label}"] = abs_delta
            out[f"scale_{metric}_rel_delta_{left_label}_to_{right_label}"] = rel_delta


def add_global_response_features(out: pd.DataFrame, merged: pd.DataFrame, metrics_used: list[str]) -> dict[str, pd.Series]:
    scale_lookup = {scale["label"]: scale for scale in SCALES}
    global_slopes: dict[str, pd.Series] = {}
    for metric in metrics_used:
        stats = merged.apply(lambda row: row_valid_points(row, scale_lookup, metric), axis=1)
        out[f"scale_{metric}_global_slope"] = stats.apply(lambda item: linear_response_stats(item[0], item[1])[0])
        out[f"scale_{metric}_global_intercept"] = stats.apply(lambda item: linear_response_stats(item[0], item[1])[1])
        out[f"scale_{metric}_global_r2"] = stats.apply(lambda item: linear_response_stats(item[0], item[1])[2])
        out[f"scale_{metric}_response_area"] = stats.apply(lambda item: response_area(item[0], item[1]))
        out[f"scale_{metric}_normalized_area"] = stats.apply(lambda item: normalized_area(item[0], item[1]))
        out[f"scale_{metric}_response_curvature"] = stats.apply(lambda item: response_curvature(item[0], item[1]))
        out[f"scale_{metric}_response_variance"] = stats.apply(lambda item: float(np.var(item[1])) if item[1].size else np.nan)
        out[f"scale_{metric}_response_range"] = stats.apply(lambda item: float(np.max(item[1]) - np.min(item[1])) if item[1].size else np.nan)
        out[f"scale_{metric}_min_scale_index"] = stats.apply(lambda item: float(item[0][int(np.argmin(item[1]))]) if item[1].size else np.nan)
        out[f"scale_{metric}_max_scale_index"] = stats.apply(lambda item: float(item[0][int(np.argmax(item[1]))]) if item[1].size else np.nan)
        global_slopes[metric] = out[f"scale_{metric}_global_slope"]
    return global_slopes


def add_monotonicity_saturation_features(out: pd.DataFrame, merged: pd.DataFrame, metrics_used: list[str]) -> None:
    scale_lookup = {scale["label"]: scale for scale in SCALES}
    for metric in metrics_used:
        stats = merged.apply(lambda row: row_valid_points(row, scale_lookup, metric), axis=1)
        out[f"scale_{metric}_monotonic_decrease_ratio"] = stats.apply(lambda item: monotonicity_stats(item[0], item[1])[0])
        out[f"scale_{metric}_monotonic_increase_ratio"] = stats.apply(lambda item: monotonicity_stats(item[0], item[1])[1])
        out[f"scale_{metric}_num_direction_changes"] = stats.apply(lambda item: monotonicity_stats(item[0], item[1])[2])
        saturation = merged.apply(lambda row: early_late_saturation(row, metric), axis=1)
        out[f"scale_{metric}_early_gain_1_5b_7b"] = saturation.apply(lambda item: item[0])
        out[f"scale_{metric}_late_gain_14b_32b"] = saturation.apply(lambda item: item[1])
        out[f"scale_{metric}_saturation_14b_32b"] = saturation.apply(lambda item: item[2])
        out[f"scale_{metric}_saturation_ratio_14b_32b"] = saturation.apply(lambda item: item[3])


def add_cross_metric_features(out: pd.DataFrame, merged: pd.DataFrame, global_slopes: dict[str, pd.Series]) -> None:
    if "ppl" in global_slopes and "loss_mean" in global_slopes:
        out["scale_ppl_loss_mean_global_slope_ratio"] = safe_divide(global_slopes["ppl"], global_slopes["loss_mean"])
    if "loss_std" in global_slopes and "loss_mean" in global_slopes:
        out["scale_loss_std_loss_mean_global_slope_ratio"] = safe_divide(global_slopes["loss_std"], global_slopes["loss_mean"])

    gap_scale_points = []
    for scale in SCALES:
        top_col = f"{scale['label']}__top_10_percent_loss_mean"
        bottom_col = f"{scale['label']}__bottom_10_percent_loss_mean"
        if top_col in merged.columns and bottom_col in merged.columns:
            out[f"scale_top_bottom_loss_gap_{scale['label']}"] = to_numeric_series(merged[top_col]) - to_numeric_series(merged[bottom_col])
            gap_scale_points.append(scale["label"])
    if not gap_scale_points:
        return

    scale_lookup = {scale["label"]: scale for scale in SCALES}
    gap_cols = [f"scale_top_bottom_loss_gap_{label}" for label in gap_scale_points if f"scale_top_bottom_loss_gap_{label}" in out.columns]
    if len(gap_cols) < 2:
        out["scale_top_bottom_loss_gap_global_slope"] = np.nan
        return

    def top_bottom_slope(row: pd.Series) -> float:
        xs = []
        ys = []
        for label in gap_scale_points:
            col = f"scale_top_bottom_loss_gap_{label}"
            value = row.get(col)
            if pd.notna(value) and np.isfinite(value):
                xs.append(scale_lookup[label]["index"])
                ys.append(float(value))
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        return linear_response_stats(xs_arr, ys_arr)[0]

    out["scale_top_bottom_loss_gap_global_slope"] = out.apply(top_bottom_slope, axis=1)


def build_scale_response_features(feature_dir: str | Path, output: str | Path) -> pd.DataFrame:
    merged, available_scales, metrics_used = load_probability_features(feature_dir)
    out = pd.DataFrame({"id": merged["id"].astype(str)}) if "id" in merged.columns else pd.DataFrame({"id": pd.Series(dtype=str)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        add_pairwise_response_features(out, merged, metrics_used)
        global_slopes = add_global_response_features(out, merged, metrics_used)
        add_monotonicity_saturation_features(out, merged, metrics_used)
        add_cross_metric_features(out, merged, global_slopes)
    out = sanitize_numeric_frame(out)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    manifest = {
        "available_scales": [scale["label"] for scale in available_scales],
        "metrics_used": metrics_used,
        "feature_count": int(len(out.columns) - 1),
        "feature_names": [col for col in out.columns if col != "id"],
    }
    save_json(manifest, output.with_name("scale_response_manifest.json"))
    print(f"Saved scale response features: {output}")
    print(f"Final rows: {len(out)}")
    print(f"Final feature count: {len(out.columns) - 1}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-scale probability response features.")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--output", default="features/scale_response_features.csv")
    args = parser.parse_args()
    build_scale_response_features(args.feature_dir, args.output)


if __name__ == "__main__":
    main()
