"""Project configuration for MS-ME-Detect."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = PROJECT_ROOT / "features"
RESULT_DIR = PROJECT_ROOT / "results"

DATA_PATH = DATA_DIR / "dataset.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MAX_LENGTH = 1024
DEFAULT_SMALL_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_MEDIUM_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_LARGE_MODEL = "Qwen/Qwen2.5-14B"
DEFAULT_XL_MODEL = "Qwen/Qwen2.5-32B"
DEFAULT_INSTRUCT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_STRONG_INSTRUCT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

MODEL_ROOT = os.environ.get("MODEL_ROOT", "/vepfs-mlp2/queue010/20252203113/models")

MODEL_REGISTRY = {
    "small": {
        "repo_id": DEFAULT_SMALL_MODEL,
        "local_dir": f"{MODEL_ROOT}/Qwen2.5-1.5B",
        "type": "base",
    },
    "medium": {
        "repo_id": DEFAULT_MEDIUM_MODEL,
        "local_dir": f"{MODEL_ROOT}/Qwen2.5-7B",
        "type": "base",
    },
    "large": {
        "repo_id": DEFAULT_LARGE_MODEL,
        "local_dir": f"{MODEL_ROOT}/Qwen2.5-14B",
        "type": "base",
    },
    "xl": {
        "repo_id": DEFAULT_XL_MODEL,
        "local_dir": f"{MODEL_ROOT}/Qwen2.5-32B",
        "type": "base",
    },
    "instruct_large": {
        "repo_id": DEFAULT_INSTRUCT_MODEL,
        "local_dir": f"{MODEL_ROOT}/Qwen2.5-14B-Instruct",
        "type": "instruct",
    },
    "instruct_xl": {
        "repo_id": DEFAULT_STRONG_INSTRUCT_MODEL,
        "local_dir": f"{MODEL_ROOT}/Qwen2.5-32B-Instruct",
        "type": "instruct",
    },
}

MODEL_KEY_PREFIX = {
    "small": "qwen25_1_5b",
    "medium": "qwen25_7b",
    "large": "qwen25_14b",
    "xl": "qwen25_32b",
    "instruct_large": "qwen25_14b_instruct",
    "instruct_xl": "qwen25_32b_instruct",
}

MODEL_FALLBACKS = [
    DEFAULT_XL_MODEL,
    DEFAULT_LARGE_MODEL,
    DEFAULT_MEDIUM_MODEL,
    DEFAULT_SMALL_MODEL,
    "Qwen/Qwen2.5-0.5B",
]

USE_4BIT = False
DTYPE = "bfloat16"
RANDOM_STATE = 42
TEST_SIZE = 0.2

METADATA_COLUMNS = ["id", "label", "type", "source", "topic", "text"]


def ensure_dirs() -> None:
    """Create standard output directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def get_model_entry(model_key: str) -> dict:
    """Return a model registry entry by key."""
    if model_key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {valid}")
    return MODEL_REGISTRY[model_key]


def get_model_local_path(model_key: str) -> str:
    """Return the configured local model path for a registry key."""
    return get_model_entry(model_key)["local_dir"]


def is_local_model_ready(path: str | Path) -> bool:
    """Lightweight local model validation for config/tokenizer/weights."""
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_tokenizer = any((path / name).exists() for name in ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"])
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return has_config and has_tokenizer and has_weights


def resolve_model_path(
    model_or_key: str,
    *,
    auto_download: bool = False,
    online: bool = False,
    model_root: str | None = None,
) -> str:
    """Resolve a model key, local path, or repo id to a loadable model reference.

    Registry keys prefer local paths. Missing registry models raise unless online
    access or explicit auto-download behavior is requested by the caller.
    """
    if model_root:
        for entry in MODEL_REGISTRY.values():
            entry["local_dir"] = str(Path(model_root) / Path(entry["local_dir"]).name)

    if model_or_key in MODEL_REGISTRY:
        entry = get_model_entry(model_or_key)
        local_dir = entry["local_dir"]
        if is_local_model_ready(local_dir):
            return local_dir
        if auto_download or online:
            return entry["repo_id"]
        raise FileNotFoundError(
            "Local model not found or incomplete. "
            f"Run: python scripts/download_models.py --models {model_or_key}"
        )

    candidate = Path(model_or_key)
    if candidate.exists():
        return str(candidate)

    if "/" in model_or_key:
        if online or auto_download:
            return model_or_key
        raise FileNotFoundError(
            f"Model '{model_or_key}' is not a local path. "
            "Use --auto_download or omit --local_files_only to allow Transformers/Hugging Face access, "
            "or run scripts/download_models.py and use --model_key."
        )

    return model_or_key
