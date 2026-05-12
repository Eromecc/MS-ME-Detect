"""Merge feature CSV files into a trainable matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
    from .preprocess import load_dataset
    from .utils import write_csv
except ImportError:
    import config
    from preprocess import load_dataset
    from utils import write_csv


def infer_feature_group(column: str) -> str:
    if column.startswith("burst_"):
        return "burstiness"
    if column.startswith("qwen25_"):
        return "probability"
    if column.startswith("scale_"):
        return "scale_response"
    if column.startswith("bino_"):
        return "binoculars_style"
    if column.startswith("struct_"):
        return "structure"
    if column.startswith("pert_"):
        return "perturbation"
    return "other"


def merge_features(data_path: str | Path, feature_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    base = load_dataset(data_path)
    keep = ["id", "label", "type", "source", "topic", "text"]
    merged = base[keep].copy()
    for path in sorted(Path(feature_dir).glob("*.csv")):
        if path.name == Path(output_path).name:
            continue
        frame = pd.read_csv(path)
        if "id" not in frame.columns:
            continue
        frame["id"] = frame["id"].astype(str)
        drop_meta = [c for c in ["label", "type", "source", "topic", "text"] if c in frame.columns]
        frame = frame.drop(columns=drop_meta)
        merged = merged.merge(frame, on="id", how="left")
    metadata = [c for c in keep if c in merged.columns]
    feature_cols = [c for c in merged.columns if c not in metadata]
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    numeric_cols = merged[feature_cols].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        med = merged[col].median() if merged[col].notna().any() else 0.0
        merged[col] = merged[col].fillna(0.0 if pd.isna(med) else med)
    non_numeric = [c for c in feature_cols if c not in numeric_cols]
    merged = merged.drop(columns=non_numeric)
    write_csv(merged, output_path)
    groups = [{"feature": c, "group": infer_feature_group(c)} for c in merged.columns if c not in metadata]
    write_csv(pd.DataFrame(groups), Path(output_path).with_name("feature_groups.csv"))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.DATA_PATH))
    parser.add_argument("--feature_dir", default=str(config.FEATURE_DIR))
    parser.add_argument("--output", default=str(config.FEATURE_DIR / "all_features.csv"))
    args = parser.parse_args()
    config.ensure_dirs()
    merge_features(args.data, args.feature_dir, args.output)


if __name__ == "__main__":
    main()
