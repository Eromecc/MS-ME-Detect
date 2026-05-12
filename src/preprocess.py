"""Dataset loading, cleaning, sentence splitting, and train/test splitting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from . import config
except ImportError:
    import config


REQUIRED_COLUMNS = ["id", "text", "label", "type", "source", "topic"]


def clean_text(text: str) -> str:
    """Normalize excessive whitespace while preserving Chinese/English punctuation."""
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？!?；;.!?])\s*", text)
    return [p.strip() for p in pieces if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def create_demo_dataset(path: str | Path) -> pd.DataFrame:
    """Create a tiny Chinese/English mixed demo dataset."""
    rows = [
        (1, "今天下班后我去菜市场买了青菜和豆腐，路上遇到老同学，我们聊了十分钟。", 0, "Human", "diary", "daily"),
        (2, "综上所述，人工智能技术在教育领域具有重要意义，不仅提升效率，而且为未来研究提供了新的思路。", 1, "AI-generated", "Qwen", "academic"),
        (3, "The meeting was moved to Friday because two team members had field interviews on Thursday.", 0, "Human", "office", "report"),
        (4, "It is worth noting that this framework significantly improves robustness and provides valuable references for future exploration.", 1, "AI-generated", "ChatGPT", "academic"),
        (5, "这篇报道采访了三位居民，他们对旧小区改造的时间表和停车方案仍有疑问。", 0, "Human", "article", "news"),
        (6, "随着数字化转型的不断深入，企业管理模式在一定程度上呈现出系统化、智能化和协同化特征。", 1, "AI-polished", "DeepSeek", "report"),
        (7, "I wrote the first draft at midnight, so the introduction is messy but the experiment notes are complete.", 0, "Human", "student", "essay"),
        (8, "总而言之，该方法可以看出具有较强的适用性，有助于推动相关领域的高质量发展。", 1, "AI-generated", "Qwen", "essay"),
        (9, "老张说雨停以后再搬设备，因为仓库门口那段路太滑，叉车不好转弯。", 0, "Human", "conversation", "daily"),
        (10, "一方面，该策略能够显著提升资源配置效率；另一方面，也为政策优化提供参考。", 1, "AI-polished", "ChatGPT", "policy"),
        (11, "The article quotes exact budget numbers, names the contractor, and explains why the deadline changed twice.", 0, "Human", "article", "news"),
        (12, "不可忽视的是，多维度评价体系能够进一步探索复杂场景下的潜在规律。", 1, "AI-generated", "DeepSeek", "academic"),
    ]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        df = create_demo_dataset(path)
    else:
        df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in {"label"} else 0
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].copy()
    df["id"] = df["id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df[REQUIRED_COLUMNS]


def train_test_split_dataset(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataset(input_path)
    stratify = df["label"] if df["label"].nunique() > 1 and df["label"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--train", default=str(config.TRAIN_PATH))
    parser.add_argument("--test", default=str(config.TEST_PATH))
    parser.add_argument("--test_size", type=float, default=config.TEST_SIZE)
    args = parser.parse_args()
    config.ensure_dirs()
    df = load_dataset(args.input)
    df.to_csv(args.input, index=False)
    train_test_split_dataset(args.input, args.train, args.test, args.test_size, config.RANDOM_STATE)
    print(f"Prepared dataset: {len(df)} rows")


if __name__ == "__main__":
    main()
