#!/usr/bin/env python3
"""Combine numeric columns from multiple split-aligned feature roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META = {
    "id",
    "text",
    "label",
    "source_dataset",
    "language",
    "domain",
    "generator",
    "source",
    "split",
    "extra_metadata",
    "text_hash",
    "attack_type",
    "type",
    "topic",
}

CANONICAL_META = {
    ("fakespot_like", "train"): ROOT / "data/reproduction_datasets/fakespot_like_train.csv",
    ("fakespot_like", "val"): ROOT / "data/reproduction_datasets/fakespot_like_val.csv",
    ("fakespot_like", "all_samples"): ROOT / "data/test/all_samples_prepared.csv",
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]


def parse_roots(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name.strip()] = resolve(path.strip())
    return out


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c in META]].copy()


def canonical_metadata(dataset: str, split: str) -> pd.DataFrame | None:
    path = CANONICAL_META.get((dataset, split))
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "id" not in df.columns:
        return None
    return metadata_frame(df)


def combine_split(dataset: str, split: str, roots: dict[str, Path], output_root: Path) -> dict[str, object]:
    meta = canonical_metadata(dataset, split)
    merged = meta.copy() if meta is not None else None
    used = []
    for name, root in roots.items():
        path = root / dataset / split / "all_features.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "id" not in df.columns:
            continue
        if meta is None:
            keep = [c for c in df.columns if c in META]
            meta = df[keep].copy()
            merged = meta.copy()
        cols = numeric_cols(df)
        if not cols:
            continue
        part = df[["id"] + cols].copy()
        part = part.rename(columns={c: f"{name}__{c}" for c in cols})
        merged = merged.merge(part, on="id", how="left", validate="one_to_one")
        used.append(name)
    if merged is None:
        raise FileNotFoundError(f"No roots found for {dataset}/{split}")
    path = output_root / dataset / split / "all_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return {"split": split, "rows": int(len(merged)), "features": int(len(numeric_cols(merged))), "used_roots": used, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["fakespot_like"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "all_samples"])
    args = parser.parse_args()
    roots = parse_roots(args.roots)
    output_root = resolve(args.output_dir)
    manifest = {"roots": {k: str(v) for k, v in roots.items()}, "datasets": []}
    for dataset in args.datasets:
        item = {"dataset": dataset, "splits": []}
        for split in args.splits:
            item["splits"].append(combine_split(dataset, split, roots, output_root))
        manifest["datasets"].append(item)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "combined_feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
