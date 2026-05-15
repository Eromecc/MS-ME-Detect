#!/usr/bin/env python3
"""Prepare a unified English public dataset for LLM-generated text detection."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "id",
    "text",
    "label",
    "source_dataset",
    "language",
    "domain",
    "generator",
    "attack_type",
    "split",
]
GROUP_COLUMNS = ["source_dataset", "domain", "generator", "attack_type", "label"]
SUMMARY_FIELDS = ["source_dataset", "language", "domain", "generator", "attack_type", "label", "split"]
SPLIT_CHOICES = {"train", "dev", "test", "external_test"}

TEXT_COLUMNS = ["text", "content", "article", "document", "essay", "response", "generation", "output", "decoded", "completion", "body"]
LABEL_COLUMNS = ["label", "is_ai", "generated", "is_generated", "ai_generated", "machine_generated"]
SOURCE_SPLIT_COLUMNS = ["split", "set", "subset", "partition"]
DOMAIN_COLUMNS = ["domain", "topic", "category", "source", "genre"]
GENERATOR_COLUMNS = ["generator", "model", "model_name", "source_model", "llm", "engine"]
ATTACK_COLUMNS = ["attack_type", "attack", "perturbation", "transformation"]
M4_ENGLISH_HINTS = ("arxiv_", "peerread_", "reddit_", "wikihow_", "wikipedia_")


@dataclass
class BuildStats:
    empty_text_removed: int = 0
    short_text_removed: int = 0
    long_text_removed: int = 0
    duplicate_text_removed: int = 0
    invalid_label_removed: int = 0


def print_missing_source(dataset_name: str, expected_paths: list[Path]) -> None:
    print(f"[{dataset_name}] No local data files found.")
    print(f"[{dataset_name}] Looked for paths:")
    for path in expected_paths:
        print(f"  - {path}")
    print(f"[{dataset_name}] Place the raw files under one of the paths above, then rerun this script.")
    print(f"[{dataset_name}] Skipping data source: {dataset_name}")


def expected_paths_for_dataset(input_root: Path, dataset_name: str) -> list[Path]:
    base = input_root / dataset_name
    common = [
        base,
        base / "data",
        base / "train.csv",
        base / "train.jsonl",
        base / "test.csv",
        base / "test.jsonl",
        base / "data.csv",
        base / "data.jsonl",
        base / "dataset.csv",
        base / "dataset.jsonl",
        base / "metadata.csv",
        base / "metadata.jsonl",
    ]
    if dataset_name == "ghostbuster":
        common.extend(
            [
                input_root / "ghostbuster",
                input_root / "ghostbuster" / "essay",
                input_root / "ghostbuster" / "other",
                input_root / "ghostbuster" / "perturb",
                input_root / "ghostbuster" / "reuter",
                input_root / "ghostbuster" / "wp",
            ]
        )
    elif dataset_name == "m4":
        common.extend(
            [
                input_root / "m4",
                input_root / "m4" / "english",
                input_root / "m4_english",
                input_root / "m4-english",
            ]
        )
    elif dataset_name == "raid":
        common.extend(
            [
                input_root / "raid",
                input_root / "raid" / "english",
                input_root / "raid_english",
                input_root / "raid-english",
            ]
        )
    elif dataset_name == "hc3_plus":
        common.extend(
            [
                input_root / "hc3_plus",
                input_root / "hc3_plus" / "data",
                input_root / "hc3_plus" / "data" / "en",
                input_root / "chatgpt-comparison-detection-HC3-Plus-main",
            ]
        )
    return common


def discover_files(input_root: Path, dataset_name: str) -> list[Path]:
    roots = []
    if dataset_name == "m4":
        roots = [input_root / "m4", input_root / "m4_english", input_root / "m4-english", input_root / "m4" / "english"]
    elif dataset_name == "raid":
        roots = [input_root / "raid", input_root / "raid_english", input_root / "raid-english", input_root / "raid" / "english"]
    elif dataset_name == "hc3_plus":
        roots = [input_root / "hc3_plus", input_root / "hc3_plus" / "data" / "en", input_root / "chatgpt-comparison-detection-HC3-Plus-main"]
    else:
        roots = [input_root / dataset_name]

    exts = {".csv", ".tsv", ".jsonl", ".json", ".parquet", ".txt"}
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in exts:
            files.append(root)
        elif root.is_dir():
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in exts:
                    continue
                path_text = str(path).lower()
                if dataset_name == "ghostbuster":
                    if "logprobs" in path_text:
                        continue
                if dataset_name == "m4" and not any(token in path.name.lower() for token in M4_ENGLISH_HINTS):
                    continue
                if dataset_name == "hc3_plus" and "/data/en/" not in path_text and "\\data\\en\\" not in path_text:
                    continue
                if dataset_name != "ghostbuster" and path.suffix.lower() == ".txt":
                    continue
                files.append(path)
    return sorted(dict.fromkeys(files))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return pd.DataFrame({"text": [path.read_text(encoding="utf-8", errors="ignore")]})
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def sample_jsonl_records(path: Path, limit: int, seed: int) -> list[dict]:
    rng = random.Random((hash(path.name) & 0xFFFFFFFF) ^ seed)
    sample: list[dict] = []
    seen = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            if len(sample) < limit:
                sample.append(record)
                continue
            replacement = rng.randint(1, seen)
            if replacement <= limit:
                sample[replacement - 1] = record
    return sample


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def normalize_domain(value) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return "unknown"
    rules = {
        "essay": ["essay", "student", "school", "sst"],
        "creative_writing": ["creative", "story", "fiction", "poem"],
        "news": ["news", "article", "journalism"],
        "wiki": ["wiki", "wikipedia"],
        "arxiv": ["arxiv", "paper", "abstract", "academic"],
        "qa": ["qa", "q&a", "question", "answer"],
        "reddit": ["reddit", "social", "comment", "forum"],
    }
    for norm, tokens in rules.items():
        if any(token in text for token in tokens):
            return norm
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_") or "unknown"


def infer_domain_from_path(path: Path) -> str:
    path_text = str(path).lower()
    if "essay" in path_text:
        return "essay"
    if "peerread" in path_text:
        return "arxiv"
    if any(token in path_text for token in ["fiction", "creative", "story"]):
        return "creative_writing"
    if "news" in path_text:
        return "news"
    if "wikihow" in path_text:
        return "qa"
    if "wiki" in path_text or "wikipedia" in path_text:
        return "wiki"
    if "arxiv" in path_text or "abstract" in path_text:
        return "arxiv"
    if "reddit" in path_text:
        return "reddit"
    if "qa" in path_text or "question" in path_text:
        return "qa"
    return "unknown"


def normalize_generator(value, label) -> str:
    if pd.notna(label) and int(label) == 0:
        return "human"
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return "unknown"
    mapping = {
        "human": ["human", "real", "reference"],
        "chatgpt": ["chatgpt"],
        "gpt3.5": ["gpt-3.5", "gpt3.5", "turbo", "gpt_3_5"],
        "gpt4": ["gpt-4", "gpt4"],
        "davinci": ["davinci", "text-davinci"],
        "cohere": ["cohere"],
        "llama": ["llama", "llama2", "llama3"],
        "mistral": ["mistral", "mixtral"],
        "claude": ["claude"],
    }
    for norm, tokens in mapping.items():
        if any(token in text for token in tokens):
            return norm
    return re.sub(r"[^a-z0-9._-]+", "_", text).strip("_") or "unknown"


def infer_generator_from_path(path: Path, label: int | None) -> str:
    if label == 0:
        return "human"
    text = str(path).lower()
    if "chatgpt" in text:
        return "chatgpt"
    if "davinci" in text:
        return "davinci"
    if "gpt4" in text or "gpt-4" in text:
        return "gpt4"
    if "gpt3.5" in text or "gpt-3.5" in text or "turbo" in text:
        return "gpt3.5"
    if "cohere" in text:
        return "cohere"
    if "llama" in text:
        return "llama"
    if "mistral" in text:
        return "mistral"
    if "claude" in text:
        return "claude"
    return "unknown"


def normalize_attack(value) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return "none"
    mapping = {
        "none": ["none", "original", "clean", "no_attack", "human"],
        "paraphrase": ["paraphrase", "rewrite", "rephrase"],
        "synonym": ["synonym", "substitution"],
        "misspelling": ["misspelling", "typo"],
        "homoglyph": ["homoglyph"],
        "whitespace": ["whitespace", "spacing"],
        "mixed": ["mixed", "combo", "combined"],
    }
    for norm, tokens in mapping.items():
        if any(token in text for token in tokens):
            return norm
    return "unknown"


def normalize_split(value) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return None
    mapping = {
        "train": ["train", "training"],
        "dev": ["dev", "valid", "validation", "val"],
        "test": ["test", "eval", "evaluation"],
        "external_test": ["external_test", "external", "ood", "attack"],
    }
    for norm, tokens in mapping.items():
        if text == norm or text in tokens:
            return norm
    return None


def infer_split_from_path(path: Path) -> str | None:
    text = str(path).lower()
    if "train" in text:
        return "train"
    if any(token in text for token in ["val", "valid", "validation", "dev"]):
        return "dev"
    if "test" in text:
        return "test"
    return None


def infer_label(row: pd.Series, path: Path) -> int | None:
    label_col = first_existing_column(pd.DataFrame([row]), LABEL_COLUMNS)
    if label_col is not None:
        value = row[label_col]
        if pd.isna(value):
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"0", "human", "real"}:
                return 0
            if text in {"1", "ai", "generated", "machine", "llm"}:
                return 1
        try:
            numeric = int(float(value))
            return numeric if numeric in {0, 1} else None
        except Exception:
            pass

    generator_col = first_existing_column(pd.DataFrame([row]), GENERATOR_COLUMNS)
    if generator_col is not None:
        generator_value = str(row[generator_col]).strip().lower()
        if generator_value in {"human", "reference", "real"}:
            return 0
        if generator_value and generator_value != "nan":
            return 1

    joined = str(path).lower()
    if any(token in joined for token in ["human", "reference", "real"]):
        return 0
    if any(token in joined for token in ["ai", "generated", "chatgpt", "gpt", "llama", "mistral", "claude", "davinci", "cohere"]):
        return 1
    return None


def infer_text_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(df, TEXT_COLUMNS)


def infer_metadata_value(row: pd.Series, candidates: list[str]):
    col = first_existing_column(pd.DataFrame([row]), candidates)
    return row[col] if col is not None else None


def assign_default_split(df: pd.DataFrame, dataset_name: str, seed: int) -> pd.Series:
    existing = df["split"].map(normalize_split)
    if existing.notna().any():
        out = existing.copy()
    else:
        out = pd.Series([None] * len(df), index=df.index, dtype=object)

    if dataset_name == "raid":
        out = out.fillna("external_test")
        attacked = df["attack_type"].fillna("unknown") != "none"
        out.loc[attacked] = "external_test"
        return out

    missing = out.isna()
    if not missing.any():
        return out

    shuffled = df.loc[missing].sample(frac=1.0, random_state=seed).index.tolist()
    total = len(shuffled)
    train_cut = int(total * 0.8)
    dev_cut = int(total * 0.9)
    split_map = {}
    for idx, row_idx in enumerate(shuffled):
        if idx < train_cut:
            split_map[row_idx] = "train"
        elif idx < dev_cut:
            split_map[row_idx] = "dev"
        else:
            split_map[row_idx] = "test"
    out.loc[missing] = out.loc[missing].index.map(split_map.get)
    return out.fillna("train")


def sample_groups(df: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    parts = []
    for _, group in df.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        if len(group) <= max_per_group:
            parts.append(group)
        else:
            parts.append(group.sample(n=max_per_group, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def clean_dataset(df: pd.DataFrame, min_words: int, max_words: int, stats: BuildStats) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["text"] = out["text"].astype(str).str.strip()

    empty_mask = out["text"].eq("") | out["text"].str.lower().eq("nan")
    stats.empty_text_removed += int(empty_mask.sum())
    out = out.loc[~empty_mask].copy()

    out["word_count"] = out["text"].str.split().map(len)
    short_mask = out["word_count"] < min_words
    long_mask = out["word_count"] > max_words
    stats.short_text_removed += int(short_mask.sum())
    stats.long_text_removed += int(long_mask.sum())
    out = out.loc[~short_mask & ~long_mask].copy()

    invalid_label_mask = ~out["label"].isin([0, 1])
    stats.invalid_label_removed += int(invalid_label_mask.sum())
    out = out.loc[~invalid_label_mask].copy()

    dedup_mask = out.duplicated(subset=["text"], keep="first")
    stats.duplicate_text_removed += int(dedup_mask.sum())
    out = out.loc[~dedup_mask].copy()

    out = out.drop(columns=["word_count"])
    return out


def build_rows_from_files(files: list[Path], dataset_name: str, max_per_group: int, seed: int) -> pd.DataFrame:
    rows = []
    for path in files:
        if dataset_name == "m4" and path.suffix.lower() == ".jsonl":
            try:
                sampled_records = sample_jsonl_records(path, max_per_group, seed)
            except Exception as exc:
                print(f"[{dataset_name}] Failed to sample {path}: {exc}")
                continue
            if not sampled_records:
                continue
            for idx, row_dict in enumerate(sampled_records):
                row = pd.Series(row_dict)
                domain = normalize_domain(infer_metadata_value(row, DOMAIN_COLUMNS) or infer_domain_from_path(path) or path.parent.name)
                split = normalize_split(infer_metadata_value(row, SOURCE_SPLIT_COLUMNS))
                source_model = infer_metadata_value(row, GENERATOR_COLUMNS) or row.get("model")
                source_id = row.get("source_id", idx)
                source_suffix = str(source_id).strip() if pd.notna(source_id) else str(idx)
                rows.append(
                    {
                        "id": f"m4_{path.stem}_{source_suffix}_human",
                        "text": row.get("human_text", ""),
                        "label": 0,
                        "source_dataset": "m4",
                        "language": "en",
                        "domain": domain,
                        "generator": "human",
                        "attack_type": "none",
                        "split": split,
                    }
                )
                rows.append(
                    {
                        "id": f"m4_{path.stem}_{source_suffix}_machine",
                        "text": row.get("machine_text", ""),
                        "label": 1,
                        "source_dataset": "m4",
                        "language": "en",
                        "domain": domain,
                        "generator": normalize_generator(source_model or path.stem, 1),
                        "attack_type": "none",
                        "split": split,
                    }
                )
            continue
        try:
            frame = read_table(path)
        except Exception as exc:
            print(f"[{dataset_name}] Failed to read {path}: {exc}")
            continue
        if frame.empty:
            continue
        text_col = infer_text_column(frame)
        if text_col is None:
            print(f"[{dataset_name}] Skipping {path}; no text column found.")
            continue
        path_text = str(path).lower()
        for idx, row in frame.iterrows():
            label = infer_label(row, path)
            record = {
                "id": f"{dataset_name}_{path.stem}_{idx}",
                "text": row.get(text_col, ""),
                "label": label,
                "source_dataset": dataset_name,
                "language": "en",
                "domain": normalize_domain(infer_metadata_value(row, DOMAIN_COLUMNS) or infer_domain_from_path(path) or path.parent.name),
                "generator": normalize_generator(
                    infer_metadata_value(row, GENERATOR_COLUMNS) or infer_generator_from_path(path, label) or path.parent.name,
                    label if label is not None else 1,
                ),
                "attack_type": normalize_attack(infer_metadata_value(row, ATTACK_COLUMNS) or path.parent.name),
                "split": normalize_split(infer_metadata_value(row, SOURCE_SPLIT_COLUMNS)) or infer_split_from_path(path),
            }
            if dataset_name == "raid" and record["attack_type"] == "none" and "attack" in path_text:
                record["attack_type"] = "unknown"
            rows.append(record)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def load_ghostbuster(input_root: Path, max_per_group: int, seed: int) -> pd.DataFrame:
    expected = expected_paths_for_dataset(input_root, "ghostbuster")
    files = discover_files(input_root, "ghostbuster")
    if not files:
        print_missing_source("ghostbuster", expected)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = build_rows_from_files(files, "ghostbuster", max_per_group, seed)
    if not df.empty:
        df["split"] = assign_default_split(df, "ghostbuster", seed)
        df = sample_groups(df, max_per_group, seed)
    return df


def load_m4_english(input_root: Path, max_per_group: int, seed: int) -> pd.DataFrame:
    expected = expected_paths_for_dataset(input_root, "m4")
    files = discover_files(input_root, "m4")
    if not files:
        print_missing_source("m4", expected)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = build_rows_from_files(files, "m4", max_per_group, seed)
    if not df.empty:
        df["split"] = assign_default_split(df, "m4", seed)
        df = sample_groups(df, max_per_group, seed)
    return df


def load_raid_english(input_root: Path, max_per_group: int, seed: int) -> pd.DataFrame:
    expected = expected_paths_for_dataset(input_root, "raid")
    files = discover_files(input_root, "raid")
    if not files:
        print_missing_source("raid", expected)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = build_rows_from_files(files, "raid", max_per_group, seed)
    if not df.empty:
        df["split"] = assign_default_split(df, "raid", seed)
        df = sample_groups(df, max_per_group, seed)
    return df


def load_hc3_plus_english(input_root: Path, max_per_group: int, seed: int) -> pd.DataFrame:
    expected = expected_paths_for_dataset(input_root, "hc3_plus")
    files = discover_files(input_root, "hc3_plus")
    if not files:
        print_missing_source("hc3_plus", expected)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = build_rows_from_files(files, "hc3_plus", max_per_group, seed)
    if not df.empty:
        df["source_dataset"] = "hc3_plus"
        df["language"] = "en"
        df["attack_type"] = "none"
        df["domain"] = df["domain"].replace({"unknown": "qa"})
        df["generator"] = df.apply(
            lambda row: "human"
            if row["label"] == 0
            else ("chatgpt" if str(row["generator"]).lower() in {"chatgpt", "gpt3.5", "gpt4"} else "unknown_ai"),
            axis=1,
        )
        df["split"] = assign_default_split(df, "hc3_plus", seed)
        df = sample_groups(df, max_per_group, seed)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in SUMMARY_FIELDS:
        counts = df[field].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"field": field, "value": value, "count": int(count)})
    return pd.DataFrame(rows, columns=["field", "value", "count"])


def write_quality_report(path: Path, stats: BuildStats, skipped_sources: list[str], final_rows: int) -> None:
    lines = [
        "English public dataset quality report",
        f"Final rows kept: {final_rows}",
        f"Empty text removed: {stats.empty_text_removed}",
        f"Too short removed: {stats.short_text_removed}",
        f"Too long removed: {stats.long_text_removed}",
        f"Duplicate text removed: {stats.duplicate_text_removed}",
        f"Invalid label removed: {stats.invalid_label_removed}",
        f"Skipped data sources: {', '.join(skipped_sources) if skipped_sources else 'none'}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare unified English public datasets for LLM text detection.")
    parser.add_argument("--dataset", choices=["ghostbuster", "m4", "raid", "hc3_plus", "all"], default="all")
    parser.add_argument("--input_root", default="data/raw")
    parser.add_argument("--output", default="data/dataset_english_v1.csv")
    parser.add_argument("--max_per_group", type=int, default=1000)
    parser.add_argument("--min_words", type=int, default=50)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaders = {
        "ghostbuster": load_ghostbuster,
        "m4": load_m4_english,
        "raid": load_raid_english,
        "hc3_plus": load_hc3_plus_english,
    }
    selected = list(loaders) if args.dataset == "all" else [args.dataset]

    frames = []
    skipped_sources = []
    source_counts: dict[str, int] = {}
    for dataset_name in selected:
        df = loaders[dataset_name](input_root, args.max_per_group, args.seed)
        if df.empty:
            skipped_sources.append(dataset_name)
        else:
            source_counts[dataset_name] = int(len(df))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    stats = BuildStats()
    merged = clean_dataset(merged, min_words=args.min_words, max_words=args.max_words, stats=stats)

    if not merged.empty:
        merged = sample_groups(merged, args.max_per_group, args.seed)
        merged["id"] = [f"eng_v1_{idx:08d}" for idx in range(1, len(merged) + 1)]
        merged["split"] = merged["split"].map(normalize_split).fillna("train")
        merged["language"] = "en"
        merged = merged[OUTPUT_COLUMNS].reset_index(drop=True)
    else:
        merged = pd.DataFrame(columns=OUTPUT_COLUMNS)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    quality_path = output_path.with_name(output_path.stem + "_quality_report.txt")

    merged.to_csv(output_path, index=False)
    build_summary(merged).to_csv(summary_path, index=False)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output": str(output_path),
        "data_sources": selected,
        "filter_rules": {
            "drop_empty_text": True,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "drop_duplicate_text": True,
            "allowed_labels": [0, 1],
        },
        "max_per_group": args.max_per_group,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "final_rows": int(len(merged)),
        "samples_per_source_dataset": {
            key: int(value) for key, value in merged["source_dataset"].value_counts().to_dict().items()
        },
        "skipped_data_sources": skipped_sources,
        "expected_raw_paths": {
            dataset_name: [str(path) for path in expected_paths_for_dataset(input_root, dataset_name)] for dataset_name in selected
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_quality_report(quality_path, stats, skipped_sources, len(merged))

    print(f"Saved dataset: {output_path} ({len(merged)} rows)")
    print(f"Saved summary: {summary_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved quality report: {quality_path}")
    print(f"Skipped data sources: {', '.join(skipped_sources) if skipped_sources else 'none'}")
    print("If you want to train with this dataset later, run:")
    print("cp data/dataset_english_v1.csv data/dataset.csv")


if __name__ == "__main__":
    main()
