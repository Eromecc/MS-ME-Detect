#!/usr/bin/env python3
"""Build strict per-source train/dev/test splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCES = ["ghostbuster", "m4", "hc3_plus"]
SPLITS = ["train", "dev", "test"]


def stratified_resplit(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    labels = pd.to_numeric(df["label"], errors="coerce").astype(int)
    train_df, rest = train_test_split(df, test_size=0.30, random_state=seed, stratify=labels)
    rest_labels = pd.to_numeric(rest["label"], errors="coerce").astype(int)
    dev_df, test_df = train_test_split(rest, test_size=2 / 3, random_state=seed, stratify=rest_labels)
    out = []
    for split, part in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        tmp = part.copy()
        tmp["split"] = split
        out.append(tmp)
    return pd.concat(out, ignore_index=True)


def source_has_complete_original_splits(df: pd.DataFrame) -> bool:
    if "split" not in df.columns:
        return False
    values = set(df["split"].dropna().astype(str).str.lower())
    return set(SPLITS).issubset(values)


def id_leakage_report(df: pd.DataFrame) -> dict:
    by_id = df.groupby("id")["split"].nunique()
    leaking_ids = by_id[by_id > 1].index.astype(str).tolist()
    return {"leaking_id_count": len(leaking_ids), "leaking_ids": leaking_ids[:100]}


def summary_rows(df: pd.DataFrame, source: str) -> list[dict]:
    rows = []
    for keys, group in df.groupby(["split", "label", "domain", "generator"], dropna=False):
        split, label, domain, generator = keys
        rows.append(
            {
                "source_dataset": source,
                "split": split,
                "label": int(label) if pd.notna(label) else label,
                "domain": domain,
                "generator": generator,
                "n": int(len(group)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    if "source_dataset" not in df.columns or "label" not in df.columns or "id" not in df.columns:
        raise ValueError("input must contain id, label, and source_dataset columns")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower()

    all_summary = []
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "sources": {},
    }
    for source in SOURCES:
        source_df = df[df["source_dataset"].astype(str).eq(source)].copy()
        if source_df.empty:
            raise ValueError(f"No rows found for source_dataset={source}")
        use_original = source_has_complete_original_splits(source_df)
        if use_original:
            split_df = source_df[source_df["split"].isin(SPLITS)].copy()
            method = "original_split"
        else:
            split_df = stratified_resplit(source_df, args.seed)
            method = "stratified_70_10_20"

        leak = id_leakage_report(split_df)
        if leak["leaking_id_count"]:
            raise ValueError(f"{source} has ids in multiple splits: {leak['leaking_ids'][:5]}")
        for split in SPLITS:
            part = split_df[split_df["split"].eq(split)].copy()
            part.to_csv(output_dir / f"{source}_strict_{split}.csv", index=False)
        all_summary.extend(summary_rows(split_df, source))
        label_counts = split_df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(SPLITS, fill_value=0)
        manifest["sources"][source] = {
            "method": method,
            "rows": int(len(split_df)),
            "label_counts_by_split": {idx: {str(k): int(v) for k, v in row.items()} for idx, row in label_counts.iterrows()},
            "leakage": leak,
        }

    pd.DataFrame(all_summary).to_csv(output_dir / "split_summary.csv", index=False)
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote strict source splits to {output_dir}")


if __name__ == "__main__":
    main()
