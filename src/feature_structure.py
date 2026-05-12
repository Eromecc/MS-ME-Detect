"""Structural pattern, template phrase, POS, and information density features."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from . import config
    from .preprocess import load_dataset
    from .utils import has_package, ngrams, safe_divide, tokenize_mixed, write_csv
except ImportError:
    import config
    from preprocess import load_dataset
    from utils import has_package, ngrams, safe_divide, tokenize_mixed, write_csv


TEMPLATE_PHRASES = [
    "综上所述",
    "总而言之",
    "值得注意的是",
    "不可忽视的是",
    "随着",
    "具有重要意义",
    "提供了新的思路",
    "在一定程度上",
    "一方面",
    "另一方面",
    "不仅",
    "而且",
    "未来研究",
    "进一步探索",
    "可以看出",
    "由此可见",
    "总的来说",
    "显著提升",
    "有助于",
]
TEMPLATE_REGEXES = [re.compile("为.*?提供参考")]
TRANSITION_PHRASES = ["首先", "其次", "再次", "最后", "因此", "然而", "同时", "此外", "一方面", "另一方面", "总之"]
DOMAIN_TERMS = ["模型", "算法", "数据", "实验", "政策", "研究", "系统", "效率", "风险", "指标", "framework", "model", "data", "study"]


def count_template_phrases(text: str) -> tuple[int, int]:
    count = sum(text.count(p) for p in TEMPLATE_PHRASES)
    count += sum(len(r.findall(text)) for r in TEMPLATE_REGEXES)
    transition = sum(text.count(p) for p in TRANSITION_PHRASES)
    return count, transition


def pos_ratios(text: str, token_count: int) -> dict[str, float]:
    groups = {"noun": 0, "verb": 0, "adjective": 0, "adverb": 0, "numeral": 0, "pronoun": 0, "function_word": 0}
    if token_count <= 0:
        return {f"{k}_ratio": 0.0 for k in groups}
    if has_package("jieba.posseg"):
        import jieba.posseg as pseg

        for _, flag in pseg.cut(text):
            if flag.startswith("n"):
                groups["noun"] += 1
            elif flag.startswith("v"):
                groups["verb"] += 1
            elif flag.startswith("a"):
                groups["adjective"] += 1
            elif flag.startswith("d"):
                groups["adverb"] += 1
            elif flag.startswith("m"):
                groups["numeral"] += 1
            elif flag.startswith("r"):
                groups["pronoun"] += 1
            elif flag.startswith(("u", "p", "c", "e", "y")):
                groups["function_word"] += 1
    else:
        lower = text.lower()
        groups["pronoun"] = len(re.findall(r"\b(i|we|you|he|she|they|我|我们|你|他们)\b", lower))
        groups["numeral"] = len(re.findall(r"\d+", text))
        groups["adverb"] = len(re.findall(r"\b\w+ly\b", lower))
    return {f"{k}_ratio": safe_divide(v, token_count) for k, v in groups.items()}


def repetition_features(tokens: list[str], n: int, name: str) -> dict[str, float]:
    grams = ngrams(tokens, n)
    if not grams:
        return {f"repeated_{name}_ratio": 0.0, f"top_{name}_frequency": 0.0, f"unique_{name}_ratio": 0.0}
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return {
        f"repeated_{name}_ratio": safe_divide(repeated, len(grams)),
        f"top_{name}_frequency": float(max(counts.values())),
        f"unique_{name}_ratio": safe_divide(len(counts), len(grams)),
    }


def extract_structure_features(text: str) -> dict[str, float]:
    text = str(text or "")
    tokens = [t.lower() for t in tokenize_mixed(text)]
    token_count = len(tokens)
    template_count, transition_count = count_template_phrases(text)
    feats = {
        "template_phrase_count": float(template_count),
        "template_phrase_ratio": safe_divide(template_count, token_count),
        "transition_phrase_count": float(transition_count),
        "transition_phrase_ratio": safe_divide(transition_count, token_count),
    }
    feats.update(repetition_features(tokens, 2, "bigram"))
    feats.update(repetition_features(tokens, 3, "trigram"))
    feats.update(pos_ratios(text, token_count))
    content_ratio = feats.get("noun_ratio", 0) + feats.get("verb_ratio", 0) + feats.get("adjective_ratio", 0) + feats.get("adverb_ratio", 0)
    feats.update(
        {
            "number_density": safe_divide(len(re.findall(r"\d+(?:\.\d+)?", text)), token_count),
            "english_capitalized_entity_density": safe_divide(len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)), token_count),
            "domain_term_density": safe_divide(sum(text.lower().count(t.lower()) for t in DOMAIN_TERMS), token_count),
            "content_word_ratio": content_ratio,
        }
    )
    return feats


def build_features(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = load_dataset(input_path)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Structure"):
        feats = {f"struct_{k}": v for k, v in extract_structure_features(row["text"]).items()}
        feats["id"] = row["id"]
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--output", default=str(config.FEATURE_DIR / "structure_features.csv"))
    args = parser.parse_args()
    config.ensure_dirs()
    build_features(args.input, args.output)


if __name__ == "__main__":
    main()

