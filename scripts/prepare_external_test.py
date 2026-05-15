#!/usr/bin/env python3
"""Prepare an external test JSON into the unified CSV schema."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["id", "text", "label", "source_dataset", "language", "domain", "generator", "attack_type", "split"]
TEXT_FIELDS = ["text", "content", "input", "document", "response"]
LABEL_FIELDS = ["label", "is_ai", "class", "target"]
GENERATOR_FIELDS = ["generator", "model"]
DOMAIN_FIELDS = ["domain", "category", "source"]


def first_present(record: dict, keys: list[str]):
    for key in keys:
        if key in record:
            return record[key]
    return None


def normalize_label(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"0", "human", "false", "real"}:
        return 0
    if text in {"1", "ai", "machine", "generated", "true"}:
        return 1
    try:
        numeric = int(float(value))
        return numeric if numeric in {0, 1} else pd.NA
    except Exception:
        return pd.NA


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in ["label", "domain", "generator", "attack_type", "split"]:
        counts = df[field].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"field": field, "value": value, "count": int(count)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external all_samples.json into the unified CSV schema.")
    parser.add_argument("--input", default="data/test/all_samples.json")
    parser.add_argument("--output", default="data/test/all_samples_prepared.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    manifest_path = output_path.with_name(output_path.stem.replace("_prepared", "_prepare") + "_manifest.json")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("data", [])
    rows = []
    for idx, item in enumerate(items):
        text = first_present(item, TEXT_FIELDS)
        rows.append(
            {
                "id": f"external_all_samples_{idx:06d}",
                "text": "" if text is None else str(text).strip(),
                "label": normalize_label(first_present(item, LABEL_FIELDS)),
                "source_dataset": "external_all_samples",
                "language": "en",
                "domain": str(first_present(item, DOMAIN_FIELDS) or "unknown").strip() or "unknown",
                "generator": str(first_present(item, GENERATOR_FIELDS) or "unknown").strip() or "unknown",
                "attack_type": "unknown",
                "split": "external_test",
            }
        )
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    before = len(df)
    df = df[df["text"].astype(str).str.strip().ne("")].copy()
    removed_empty = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed_dup = before - len(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    build_summary(df).to_csv(summary_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(df)),
        "has_label": bool(df["label"].notna().any()),
        "removed_empty_text": int(removed_empty),
        "removed_duplicate_text": int(removed_dup),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved external test CSV: {output_path} ({len(df)} rows)")
    print(f"Has label: {manifest['has_label']}")


if __name__ == "__main__":
    main()
