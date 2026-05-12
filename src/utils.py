"""Shared utility helpers."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import re
import warnings
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def has_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def safe_divide(a: float, b: float, default: float = 0.0, eps: float = 1e-12) -> float:
    if b is None or abs(float(b)) < eps:
        return default
    return float(a) / (float(b) + eps)


def numeric_stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return {
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cv": safe_divide(std, mean),
    }


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize Chinese and English text, preferring jieba when present."""
    text = str(text or "")
    if has_package("jieba"):
        import jieba

        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)
    return tokens


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compression_ratio_zlib(text: str) -> float:
    raw = str(text or "").encode("utf-8")
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)


def compression_ratio_gzip(text: str) -> float:
    raw = str(text or "").encode("utf-8")
    if not raw:
        return 0.0
    return len(gzip.compress(raw)) / len(raw)


def safe_skew(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or np.nanstd(arr) == 0:
        return 0.0
    try:
        from scipy.stats import skew

        return float(skew(arr, nan_policy="omit"))
    except Exception:
        m = np.nanmean(arr)
        s = np.nanstd(arr)
        return float(np.nanmean(((arr - m) / s) ** 3)) if s else 0.0


def safe_kurtosis(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 4 or np.nanstd(arr) == 0:
        return 0.0
    try:
        from scipy.stats import kurtosis

        return float(kurtosis(arr, nan_policy="omit"))
    except Exception:
        m = np.nanmean(arr)
        s = np.nanstd(arr)
        return float(np.nanmean(((arr - m) / s) ** 4) - 3.0) if s else 0.0


def save_json(obj: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def model_safe_name(model_name: str) -> str:
    name = model_name.split("/")[-1].lower()
    name = name.replace("qwen2.5", "qwen25").replace("-", "_").replace(".", "_")
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def top_numeric_deviations(row: pd.Series, reference: pd.DataFrame, importances: pd.DataFrame | None, n: int = 5):
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    scores = []
    imp_map = {}
    if importances is not None and {"feature", "importance"}.issubset(importances.columns):
        imp_map = dict(zip(importances["feature"], importances["importance"]))
    for col in numeric_cols:
        if col not in row.index or col == "label":
            continue
        vals = reference[col].astype(float)
        std = vals.std(ddof=0)
        if not std or not math.isfinite(std):
            continue
        z = abs((float(row[col]) - vals.median()) / std)
        scores.append((z * float(imp_map.get(col, 1.0)), col, float(row[col])))
    return sorted(scores, reverse=True)[:n]
