"""Koopman-inspired DMD-lite spectral features from token loss trajectories.

This module does not use raw token IDs or token strings. It maps token-level
loss sequences to low-dimensional observables, estimates a per-text linear
operator, and exports compact spectral/dynamics summaries.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .utils import write_csv
except ImportError:
    from utils import write_csv

EPS = 1e-12
DEFAULT_RIDGE = 1e-3
MIN_TOKENS = 20


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_loss_sequences(cache_path: str | Path) -> dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    out: dict[str, np.ndarray] = {}
    opener = gzip.open if cache_path.name.endswith(".gz") else open
    with opener(cache_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            vals = np.asarray(item.get("loss_sequence", []), dtype=float)
            vals = vals[np.isfinite(vals)]
            out[str(item["id"])] = vals
    return out


def fit_loss_bins(sequences: list[np.ndarray], n_states_list: list[int] | None = None) -> dict[str, list[float]]:
    n_states_list = n_states_list or [5, 7]
    vals = np.concatenate([np.asarray(s, dtype=float)[np.isfinite(s)] for s in sequences if len(s)]) if sequences else np.asarray([])
    if vals.size == 0:
        return {str(n): [] for n in n_states_list}
    out: dict[str, list[float]] = {}
    for n in n_states_list:
        if np.nanmax(vals) == np.nanmin(vals):
            out[str(n)] = []
        else:
            out[str(n)] = [float(x) for x in np.nanquantile(vals, np.linspace(0, 1, n + 1)[1:-1])]
    return out


def states_from_bins(values: np.ndarray, bins: list[float], n_states: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=int)
    if not bins:
        return np.zeros(vals.size, dtype=int)
    return np.clip(np.searchsorted(np.asarray(bins, dtype=float), vals, side="right"), 0, n_states - 1).astype(int)


def rolling_mean(vals: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(vals)
    return s.rolling(window, min_periods=1, center=True).mean().to_numpy(dtype=float)


def rolling_std(vals: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(vals)
    return s.rolling(window, min_periods=2, center=True).std().fillna(0.0).to_numpy(dtype=float)


def raw_loss_observable(losses: np.ndarray) -> np.ndarray:
    vals = np.asarray(losses, dtype=float)
    delta = np.diff(vals, prepend=vals[0])
    z = np.vstack(
        [
            vals,
            delta,
            np.abs(delta),
            rolling_mean(vals, 5),
            rolling_std(vals, 5),
            rolling_mean(vals, 11),
            rolling_std(vals, 11),
        ]
    )
    return z


def normalized_loss_observable(losses: np.ndarray) -> np.ndarray:
    vals = np.asarray(losses, dtype=float)
    sd = float(np.std(vals))
    zloss = (vals - float(np.mean(vals))) / (sd if sd > 0 else 1.0)
    delta = np.diff(zloss, prepend=zloss[0])
    return np.vstack(
        [
            zloss,
            delta,
            np.abs(delta),
            rolling_mean(zloss, 5),
            rolling_std(zloss, 5),
            rolling_mean(zloss, 11),
            rolling_std(zloss, 11),
        ]
    )


def state_onehot_observable(losses: np.ndarray, bins: list[float], n_states: int) -> np.ndarray:
    states = states_from_bins(losses, bins, n_states)
    if states.size == 0:
        return np.zeros((n_states + 2, 0), dtype=float)
    onehot = np.eye(n_states, dtype=float)[states].T
    delta = np.diff(states.astype(float), prepend=float(states[0]))
    return np.vstack([onehot, delta, np.abs(delta)])


FEATURE_NAMES = [
    "spectral_radius",
    "eig_abs_mean",
    "eig_abs_std",
    "eig_abs_min",
    "eig_abs_max",
    "eig_abs_median",
    "eig_abs_q25",
    "eig_abs_q75",
    "eig_abs_iqr",
    "stable_eig_ratio_abs_le_1",
    "unstable_eig_ratio_abs_gt_1",
    "near_unit_circle_ratio_abs_0_9_1_1",
    "eig_abs_entropy",
    "eig_angle_mean",
    "eig_angle_std",
    "eig_angle_entropy",
    "complex_eig_ratio",
    "positive_real_eig_ratio",
    "negative_real_eig_ratio",
    "koopman_fro_norm",
    "koopman_trace",
    "koopman_det",
    "koopman_rank",
    "koopman_condition_number",
    "koopman_nuclear_norm",
    "top1_singular_value",
    "top3_singular_energy_ratio",
    "top5_singular_energy_ratio",
    "low_rank_energy_ratio",
    "one_step_reconstruction_mse",
    "one_step_reconstruction_mae",
    "normalized_reconstruction_error",
    "reconstruction_error_std",
    "mean_state_velocity",
    "std_state_velocity",
    "mean_state_acceleration",
    "std_state_acceleration",
    "token_count",
    "observable_dim",
]


def entropy_hist(values: np.ndarray, bins: int = 10, value_range: tuple[float, float] | None = None) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    counts, _ = np.histogram(vals, bins=bins, range=value_range)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(float) / total
    return float(-(p * np.log2(p)).sum())


def nan_features(prefix: str, token_count: int, observable_dim: int) -> dict[str, float]:
    out = {f"{prefix}_{name}": np.nan for name in FEATURE_NAMES}
    out[f"{prefix}_token_count"] = float(token_count)
    out[f"{prefix}_observable_dim"] = float(observable_dim)
    return out


def dmd_spectral_features(z: np.ndarray, prefix: str, ridge_lambda: float = DEFAULT_RIDGE, min_tokens: int = MIN_TOKENS) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    dim, t = z.shape if z.ndim == 2 else (0, 0)
    if dim == 0 or t < min_tokens:
        return nan_features(prefix, t, dim)
    x = z[:, :-1]
    y = z[:, 1:]
    try:
        gram = x @ x.T + float(ridge_lambda) * np.eye(dim)
        k = (y @ x.T) @ np.linalg.pinv(gram)
        yhat = k @ x
        eig = np.linalg.eigvals(k)
        eig_abs = np.abs(eig)
        eig_angle = np.angle(eig)
        singular = np.linalg.svd(k, compute_uv=False)
        s2 = singular**2
        total_energy = float(np.sum(s2)) + EPS
        err = y - yhat
        step_norm = np.linalg.norm(y, axis=0)
        vel = np.linalg.norm(np.diff(z, axis=1), axis=0) if t >= 2 else np.asarray([])
        acc = np.linalg.norm(np.diff(z, n=2, axis=1), axis=0) if t >= 3 else np.asarray([])
        out = {
            f"{prefix}_spectral_radius": float(np.max(eig_abs)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_mean": float(np.mean(eig_abs)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_std": float(np.std(eig_abs)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_min": float(np.min(eig_abs)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_max": float(np.max(eig_abs)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_median": float(np.median(eig_abs)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_q25": float(np.quantile(eig_abs, 0.25)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_q75": float(np.quantile(eig_abs, 0.75)) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_iqr": float(np.quantile(eig_abs, 0.75) - np.quantile(eig_abs, 0.25)) if eig_abs.size else np.nan,
            f"{prefix}_stable_eig_ratio_abs_le_1": float(np.mean(eig_abs <= 1.0 + EPS)) if eig_abs.size else np.nan,
            f"{prefix}_unstable_eig_ratio_abs_gt_1": float(np.mean(eig_abs > 1.0 + EPS)) if eig_abs.size else np.nan,
            f"{prefix}_near_unit_circle_ratio_abs_0_9_1_1": float(np.mean((eig_abs >= 0.9) & (eig_abs <= 1.1))) if eig_abs.size else np.nan,
            f"{prefix}_eig_abs_entropy": entropy_hist(eig_abs, bins=10),
            f"{prefix}_eig_angle_mean": float(np.mean(eig_angle)) if eig_angle.size else np.nan,
            f"{prefix}_eig_angle_std": float(np.std(eig_angle)) if eig_angle.size else np.nan,
            f"{prefix}_eig_angle_entropy": entropy_hist(eig_angle, bins=12, value_range=(-math.pi, math.pi)),
            f"{prefix}_complex_eig_ratio": float(np.mean(np.abs(np.imag(eig)) > 1e-8)) if eig.size else np.nan,
            f"{prefix}_positive_real_eig_ratio": float(np.mean((np.abs(np.imag(eig)) <= 1e-8) & (np.real(eig) > 0))) if eig.size else np.nan,
            f"{prefix}_negative_real_eig_ratio": float(np.mean((np.abs(np.imag(eig)) <= 1e-8) & (np.real(eig) < 0))) if eig.size else np.nan,
            f"{prefix}_koopman_fro_norm": float(np.linalg.norm(k, ord="fro")),
            f"{prefix}_koopman_trace": float(np.real(np.trace(k))),
            f"{prefix}_koopman_det": float(np.real(np.linalg.det(k))) if dim <= 12 else np.nan,
            f"{prefix}_koopman_rank": float(np.linalg.matrix_rank(k)),
            f"{prefix}_koopman_condition_number": float(np.linalg.cond(k)),
            f"{prefix}_koopman_nuclear_norm": float(np.sum(singular)),
            f"{prefix}_top1_singular_value": float(singular[0]) if singular.size else np.nan,
            f"{prefix}_top3_singular_energy_ratio": float(np.sum(s2[: min(3, len(s2))]) / total_energy),
            f"{prefix}_top5_singular_energy_ratio": float(np.sum(s2[: min(5, len(s2))]) / total_energy),
            f"{prefix}_low_rank_energy_ratio": float(np.sum(s2[: max(1, min(2, len(s2)))]) / total_energy),
            f"{prefix}_one_step_reconstruction_mse": float(np.mean(err**2)),
            f"{prefix}_one_step_reconstruction_mae": float(np.mean(np.abs(err))),
            f"{prefix}_normalized_reconstruction_error": float(np.linalg.norm(err, ord="fro") / (np.linalg.norm(y, ord="fro") + EPS)),
            f"{prefix}_reconstruction_error_std": float(np.std(np.linalg.norm(err, axis=0))),
            f"{prefix}_mean_state_velocity": float(np.mean(vel)) if vel.size else np.nan,
            f"{prefix}_std_state_velocity": float(np.std(vel)) if vel.size else np.nan,
            f"{prefix}_mean_state_acceleration": float(np.mean(acc)) if acc.size else np.nan,
            f"{prefix}_std_state_acceleration": float(np.std(acc)) if acc.size else np.nan,
            f"{prefix}_token_count": float(t),
            f"{prefix}_observable_dim": float(dim),
        }
        return {k: (np.nan if not np.isfinite(v) else float(v)) for k, v in out.items()}
    except Exception:
        return nan_features(prefix, t, dim)


def koopman_features_from_losses(
    losses: np.ndarray,
    model_name: str,
    *,
    bins: dict[str, list[float]] | None = None,
    ridge_lambda: float = DEFAULT_RIDGE,
    min_tokens: int = MIN_TOKENS,
) -> dict[str, float]:
    vals = np.asarray(losses, dtype=float)
    vals = vals[np.isfinite(vals)]
    out: dict[str, float] = {}
    mode_to_z = {
        "raw_loss_observable": raw_loss_observable(vals),
        "per_doc_normalized_observable": normalized_loss_observable(vals),
    }
    for mode, z in mode_to_z.items():
        out.update(dmd_spectral_features(z, f"koopman_{model_name}_{mode}", ridge_lambda, min_tokens))
    bins = bins or {}
    for n in [5, 7]:
        z = state_onehot_observable(vals, bins.get(str(n), []), n)
        out.update(dmd_spectral_features(z, f"koopman_{model_name}_state_onehot{n}_observable", ridge_lambda, min_tokens))
    return out


def build_koopman_features(
    cache_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    bins: dict[str, list[float]] | None = None,
    ridge_lambda: float = DEFAULT_RIDGE,
    min_tokens: int = MIN_TOKENS,
) -> pd.DataFrame:
    cache = load_loss_sequences(cache_path)
    rows = []
    short_ids = []
    for rid, losses in cache.items():
        feats = {"id": rid}
        if len(losses) < min_tokens:
            short_ids.append(rid)
        feats.update(koopman_features_from_losses(losses, model_name, bins=bins, ridge_lambda=ridge_lambda, min_tokens=min_tokens))
        rows.append(feats)
    out = pd.DataFrame(rows)
    num_cols = [c for c in out.columns if c != "id"]
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    write_csv(out, output_path)
    manifest = {
        "created_at": now(),
        "cache_path": str(cache_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "n_rows": int(len(out)),
        "n_features": int(len(num_cols)),
        "ridge_lambda": float(ridge_lambda),
        "min_tokens": int(min_tokens),
        "short_sequence_count": int(len(short_ids)),
        "short_sequence_ids_preview": short_ids[:50],
        "nan_cells": int(out[num_cols].isna().sum().sum()),
        "inf_cells": 0,
        "bins": bins or {},
    }
    Path(output_path).with_name(Path(output_path).name.replace(".csv", "_manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def multiscale_features(one5: pd.DataFrame, seven: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    one5_pref = one5.rename(columns={c: f"{c}__one5" for c in one5.columns if c != "id"})
    seven_pref = seven.rename(columns={c: f"{c}__seven" for c in seven.columns if c != "id"})
    merged = one5_pref.merge(seven_pref, on="id", how="outer")
    rows = pd.DataFrame({"id": merged["id"]})
    base_metrics = [
        "spectral_radius",
        "eig_abs_mean",
        "one_step_reconstruction_mse",
        "normalized_reconstruction_error",
        "low_rank_energy_ratio",
        "stable_eig_ratio_abs_le_1",
        "eig_angle_entropy",
    ]
    one5_cols = [c for c in one5.columns if c != "id"]
    for c1 in one5_cols:
        metric = next((m for m in base_metrics if c1.endswith("_" + m)), None)
        if metric is None:
            continue
        c7 = c1.replace("qwen25_1_5b", "qwen25_7b")
        if c7 not in seven.columns:
            continue
        m1 = c1 + "__one5"
        m7 = c7 + "__seven"
        if m1 not in merged.columns or m7 not in merged.columns:
            continue
        stem = c1.replace("koopman_qwen25_1_5b_", "")
        a = pd.to_numeric(merged[m1], errors="coerce")
        b = pd.to_numeric(merged[m7], errors="coerce")
        rows[f"koopman_scale_1_5b_7b_{stem}_delta"] = b - a
        if metric not in {"stable_eig_ratio_abs_le_1", "eig_angle_entropy"}:
            rows[f"koopman_scale_1_5b_7b_{stem}_ratio"] = b / (a.replace(0, np.nan) + EPS)
    num_cols = [c for c in rows.columns if c != "id"]
    rows[num_cols] = rows[num_cols].replace([np.inf, -np.inf], np.nan)
    write_csv(rows, output_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--bins_json")
    parser.add_argument("--ridge_lambda", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--min_tokens", type=int, default=MIN_TOKENS)
    args = parser.parse_args()
    bins = {}
    if args.bins_json:
        payload = json.loads(Path(args.bins_json).read_text())
        bins = payload.get("bins", payload)
    build_koopman_features(args.cache, args.output, model_name=args.model_name, bins=bins, ridge_lambda=args.ridge_lambda, min_tokens=args.min_tokens)


if __name__ == "__main__":
    main()
