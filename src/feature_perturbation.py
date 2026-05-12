"""Optional perturbation-stability features."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import config
    from .preprocess import load_dataset, split_sentences
    from .utils import compression_ratio_zlib, has_package, safe_divide, tokenize_mixed, warn, write_csv
except ImportError:
    import config
    from preprocess import load_dataset, split_sentences
    from utils import compression_ratio_zlib, has_package, safe_divide, tokenize_mixed, warn, write_csv


PHRASE_REPLACEMENTS = {
    "综上所述": "",
    "总而言之": "",
    "值得注意的是": "需要注意的是",
    "不可忽视的是": "需要看到",
    "具有重要意义": "有意义",
    "显著提升": "提升",
    "进一步探索": "继续研究",
    "It is worth noting that": "",
    "significantly improves": "improves",
}
REDUNDANT_ADVERBS = ["非常", "显著", "极其", "不断", "充分", "clearly", "significantly", "notably"]


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。.!！?？]{2,}", "。", text)
    return text.strip()


def rule_based_perturbations(text: str) -> list[str]:
    variants = []
    base = str(text or "")
    t = base
    for src, dst in PHRASE_REPLACEMENTS.items():
        t = t.replace(src, dst)
    variants.append(normalize_punctuation(t))
    t2 = base
    for adv in REDUNDANT_ADVERBS:
        t2 = t2.replace(adv, "")
    variants.append(normalize_punctuation(t2))
    sentences = split_sentences(base)
    if len(sentences) > 1:
        variants.append("".join(sentences[: max(1, int(len(sentences) * 0.8))]))
    clauses = re.split(r"([，,；;])", base)
    if len(clauses) >= 5:
        clauses[0], clauses[2] = clauses[2], clauses[0]
        variants.append(normalize_punctuation("".join(clauses)))
    variants.append(normalize_punctuation(base))
    return [v for v in dict.fromkeys(variants) if v]


def llm_perturbations(text: str, model_name: str, dtype: str, max_length: int) -> list[str]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if dtype == "bfloat16" else "auto", device_map="auto", trust_remote_code=True)
        prompts = [
            "请在保持语义不变的前提下，将以下文本改写得更正式：",
            "请在保持语义不变的前提下，将以下文本改写得更简洁：",
            "请在保持语义不变的前提下，将以下文本改写得更自然、更像人工写作：",
        ]
        outs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt + "\n" + text}]
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            outs.append(tokenizer.decode(ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip())
        return [o for o in outs if o]
    except Exception as exc:
        warn(f"LLM perturbation unavailable, falling back to rules: {exc}")
        return []


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize_mixed(a)), set(tokenize_mixed(b))
    if not ta and not tb:
        return 1.0
    return safe_divide(len(ta & tb), len(ta | tb))


def semantic_similarities(text: str, variants: list[str]) -> list[float]:
    if not has_package("sentence_transformers"):
        return []
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        emb = model.encode([text] + variants, normalize_embeddings=True)
        return [float(np.dot(emb[0], e)) for e in emb[1:]]
    except Exception as exc:
        warn(f"Semantic similarity skipped: {exc}")
        return []


def extract_features(text: str, mode: str = "rule", model_name: str = config.DEFAULT_INSTRUCT_MODEL, dtype: str = "bfloat16", max_length: int = 1024) -> dict[str, float]:
    variants = []
    if mode == "llm":
        variants = llm_perturbations(text, model_name, dtype, max_length)
    variants.extend(rule_based_perturbations(text))
    variants = [v for v in dict.fromkeys(variants) if v and v != text]
    base_len = max(1, len(str(text)))
    base_comp = compression_ratio_zlib(text)
    deltas = [(len(v) - base_len) / base_len for v in variants]
    comp_deltas = [compression_ratio_zlib(v) - base_comp for v in variants]
    jaccards = [jaccard(text, v) for v in variants]
    sem = semantic_similarities(text, variants) if variants else []
    return {
        "perturbation_count": float(len(variants)),
        "avg_length_delta": float(np.mean(deltas)) if deltas else 0.0,
        "std_length_delta": float(np.std(deltas)) if deltas else 0.0,
        "avg_compression_ratio_delta": float(np.mean(comp_deltas)) if comp_deltas else 0.0,
        "avg_jaccard_similarity": float(np.mean(jaccards)) if jaccards else 0.0,
        "std_jaccard_similarity": float(np.std(jaccards)) if jaccards else 0.0,
        "semantic_similarity_mean": float(np.mean(sem)) if sem else 0.0,
        "semantic_similarity_std": float(np.std(sem)) if sem else 0.0,
    }


def build_features(input_path: str | Path, output_path: str | Path, mode: str = "rule", model_name: str = config.DEFAULT_INSTRUCT_MODEL, dtype: str = "bfloat16", max_length: int = 1024):
    df = load_dataset(input_path)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Perturbation"):
        feats = {f"pert_{k}": v for k, v in extract_features(row["text"], mode, model_name, dtype, max_length).items()}
        feats["id"] = row["id"]
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--output", default=str(config.FEATURE_DIR / "perturbation_features.csv"))
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--model", default=config.DEFAULT_INSTRUCT_MODEL)
    parser.add_argument("--dtype", default=config.DTYPE)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    args = parser.parse_args()
    config.ensure_dirs()
    build_features(args.input, args.output, args.mode, args.model, args.dtype, args.max_length)


if __name__ == "__main__":
    main()

