#!/usr/bin/env python3
"""Build per-dataset training CSVs from the unified English dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["id", "text", "label", "source_dataset", "language", "domain", "generator", "attack_type", "split"]


def build_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for train_set, df in datasets.items():
        rows.append({"train_set": train_set, "group": "total", "key": "rows", "count": int(len(df))})
        for field in ["label", "domain", "generator", "source_dataset", "split"]:
            counts = df[field].value_counts(dropna=False)
            for value, count in counts.items():
                rows.append({"train_set": train_set, "group": field, "key": str(value), "count": int(count)})
    return pd.DataFrame(rows)


def write_dataset(df: pd.DataFrame, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(path, index=False)
    labels = sorted(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int).unique().tolist())
    if labels != [0, 1]:
        print(f"Warning: {name} has label set {labels}; downstream training should skip it.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-source training CSVs from the unified English dataset.")
    parser.add_argument("--input", default="data/dataset_english_v1.csv")
    parser.add_argument("--output_dir", default="data/train_sets")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    datasets = {
        "ghostbuster": df[df["source_dataset"] == "ghostbuster"].copy(),
        "m4": df[df["source_dataset"] == "m4"].copy(),
        "hc3_plus": df[df["source_dataset"] == "hc3_plus"].copy(),
        "combined_public": df[df["source_dataset"].isin(["ghostbuster", "m4", "hc3_plus"])].copy(),
    }
    for name, part in datasets.items():
        write_dataset(part, output_dir / f"{name}_train.csv", name)
        print(f"Saved {name}: {len(part)} rows")

    build_summary(datasets).to_csv(output_dir / "train_sets_summary.csv", index=False)
    print(f"Saved summary: {output_dir / 'train_sets_summary.csv'}")


if __name__ == "__main__":
    main()
