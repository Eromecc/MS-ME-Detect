"""Statistical burstiness and regularity features."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import config
    from .preprocess import load_dataset, split_paragraphs, split_sentences
    from .utils import compression_ratio_gzip, compression_ratio_zlib, ngrams, numeric_stats, safe_divide, tokenize_mixed, write_csv
except ImportError:
    import config
    from preprocess import load_dataset, split_paragraphs, split_sentences
    from utils import compression_ratio_gzip, compression_ratio_zlib, ngrams, numeric_stats, safe_divide, tokenize_mixed, write_csv


def zipf_deviation(tokens: list[str]) -> float:
    counts = Counter(tokens)
    if len(counts) < 3:
        return 0.0
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(freqs)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    return float(np.mean(np.abs(y - fitted)))


def extract_burstiness_features(text: str) -> dict[str, float]:
    text = str(text or "")
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    tokens = tokenize_mixed(text)
    words = [t.lower() for t in tokens if re.search(r"[\w\u4e00-\u9fff]", t)]
    sent_lengths = [len(tokenize_mixed(s)) for s in sentences] or [0]
    para_lengths = [len(tokenize_mixed(p)) for p in paragraphs] or [0]
    sent_stats = numeric_stats(sent_lengths)
    para_stats = numeric_stats(para_lengths)
    char_count = len(text)
    punct_count = len(re.findall(r"[，,。.!！?？；;：:、]", text))
    repeated_chars = sum(max(0, c - 1) for c in Counter(text).values() if not c == " ")
    repeated_words = sum(max(0, c - 1) for c in Counter(words).values())
    return {
        "text_length": float(len(text)),
        "char_count": float(char_count),
        "word_count": float(len(words)),
        "sentence_count": float(len(sentences)),
        "avg_sentence_length": sent_stats["mean"],
        "std_sentence_length": sent_stats["std"],
        "sentence_length_cv": sent_stats["cv"],
        "max_sentence_length": sent_stats["max"],
        "min_sentence_length": sent_stats["min"],
        "punctuation_ratio": safe_divide(punct_count, char_count),
        "comma_ratio": safe_divide(len(re.findall(r"[，,、]", text)), char_count),
        "period_ratio": safe_divide(len(re.findall(r"[。.!]", text)), char_count),
        "question_ratio": safe_divide(len(re.findall(r"[?？]", text)), char_count),
        "exclamation_ratio": safe_divide(len(re.findall(r"[!！]", text)), char_count),
        "paragraph_count": float(len(paragraphs)),
        "avg_paragraph_length": para_stats["mean"],
        "std_paragraph_length": para_stats["std"],
        "type_token_ratio": safe_divide(len(set(words)), len(words)),
        "repeated_char_ratio": safe_divide(repeated_chars, max(1, char_count)),
        "repeated_word_ratio": safe_divide(repeated_words, max(1, len(words))),
        "compression_ratio_zlib": compression_ratio_zlib(text),
        "compression_ratio_gzip": compression_ratio_gzip(text),
        "zipf_deviation_score": zipf_deviation(words),
    }


def build_features(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = load_dataset(input_path)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Burstiness"):
        feats = extract_burstiness_features(row["text"])
        feats = {f"burst_{k}": v for k, v in feats.items()}
        feats["id"] = row["id"]
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--output", default=str(config.FEATURE_DIR / "burstiness_features.csv"))
    args = parser.parse_args()
    config.ensure_dirs()
    build_features(args.input, args.output)


if __name__ == "__main__":
    main()

