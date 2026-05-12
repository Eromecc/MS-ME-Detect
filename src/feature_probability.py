"""Qwen2.5 Base model probability and token-level loss features."""

from __future__ import annotations

import argparse
import gc
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import config
    from .preprocess import load_dataset
    from .utils import model_safe_name, safe_divide, safe_kurtosis, safe_skew, warn, write_csv
except ImportError:
    import config
    from preprocess import load_dataset
    from utils import model_safe_name, safe_divide, safe_kurtosis, safe_skew, warn, write_csv


def resolve_dtype(dtype: str):
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return "auto"


def load_causal_lm(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str | None = None,
    load_4bit: bool = False,
    local_files_only: bool = False,
):
    """Lazy-load a Hugging Face causal LM and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "instruct" in str(model_name).lower():
        warn("Warning: Instruct models are not recommended for PPL features. Use Qwen2.5 Base models instead.")

    kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=resolve_dtype(dtype))
            kwargs["device_map"] = device_map or "auto"
        except Exception as exc:
            warn(f"4-bit quantization unavailable, loading without it: {exc}")
    else:
        kwargs["torch_dtype"] = resolve_dtype(dtype)
        if device_map:
            kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device_map and not load_4bit:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    model.eval()
    return tokenizer, model


def download_model_key(model_key: str, model_root: str | None = None) -> None:
    """Explicitly download one configured model key via the project script."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
    cmd = [sys.executable, str(script), "--models", model_key]
    if model_root:
        cmd.extend(["--model_root", model_root])
    subprocess.run(cmd, check=True)


def resolve_requested_model(
    model: str | None,
    model_key: str | None,
    *,
    local_files_only: bool = False,
    auto_download: bool = False,
    model_root: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve model CLI inputs to a load reference, feature prefix, and key."""
    if model_key:
        local_path = config.get_model_local_path(model_key)
        if auto_download and not config.is_local_model_ready(local_path):
            download_model_key(model_key, model_root)
        if not config.is_local_model_ready(local_path if model_root is None else str(Path(model_root) / Path(local_path).name)):
            raise FileNotFoundError(
                "Local model not found. "
                f"Run: python scripts/download_models.py --models {model_key}"
            )
        resolved = config.resolve_model_path(model_key, auto_download=False, online=False, model_root=model_root)
        return resolved, config.MODEL_KEY_PREFIX.get(model_key, model_safe_name(resolved)), model_key

    requested = model or config.DEFAULT_SMALL_MODEL
    if "instruct" in requested.lower():
        warn("Warning: Instruct models are not recommended for PPL features. Use Qwen2.5 Base models instead.")
    resolved = config.resolve_model_path(requested, auto_download=auto_download, online=not local_files_only, model_root=model_root)
    return resolved, model_safe_name(requested), None


def empty_feature_row(prefix: str) -> dict[str, float]:
    names = [
        "ppl",
        "loss_mean",
        "loss_std",
        "loss_cv",
        "loss_min",
        "loss_max",
        "loss_range",
        "loss_skewness",
        "loss_kurtosis",
        "top_10_percent_loss_mean",
        "bottom_10_percent_loss_mean",
        "token_count_used",
    ]
    return {f"{prefix}_{name}": np.nan for name in names}


def token_loss_features(text: str, tokenizer, model, max_length: int = 1024, prefix: str = "model") -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if not str(text).strip():
        return empty_feature_row(prefix)
    device = next(model.parameters()).device
    encoded = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if input_ids.shape[1] < 2:
        return empty_feature_row(prefix)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    vals = losses.detach().float().cpu().numpy()
    if vals.size == 0:
        return empty_feature_row(prefix)
    vals_sorted = np.sort(vals)
    k = max(1, int(math.ceil(vals.size * 0.10)))
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    return {
        f"{prefix}_ppl": float(np.exp(min(mean, 50))),
        f"{prefix}_loss_mean": mean,
        f"{prefix}_loss_std": std,
        f"{prefix}_loss_cv": safe_divide(std, mean),
        f"{prefix}_loss_min": float(vals.min()),
        f"{prefix}_loss_max": float(vals.max()),
        f"{prefix}_loss_range": float(vals.max() - vals.min()),
        f"{prefix}_loss_skewness": safe_skew(vals),
        f"{prefix}_loss_kurtosis": safe_kurtosis(vals),
        f"{prefix}_top_10_percent_loss_mean": float(vals_sorted[-k:].mean()),
        f"{prefix}_bottom_10_percent_loss_mean": float(vals_sorted[:k].mean()),
        f"{prefix}_token_count_used": float(vals.size),
    }


def build_probability_features(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str | None = None,
    dtype: str = "bfloat16",
    max_length: int = 1024,
    device_map: str | None = None,
    load_4bit: bool = False,
    allow_fallback: bool = False,
    model_key: str | None = None,
    local_files_only: bool = False,
    auto_download: bool = False,
    model_root: str | None = None,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    load_name, prefix, resolved_key = resolve_requested_model(
        model_name,
        model_key,
        local_files_only=local_files_only,
        auto_download=auto_download,
        model_root=model_root,
    )
    tokenizer = model = None
    loaded_name = load_name
    candidates = [load_name]
    if allow_fallback:
        candidates += [m for m in config.MODEL_FALLBACKS if m != load_name]
    for candidate in candidates:
        try:
            tokenizer, model = load_causal_lm(
                candidate,
                dtype=dtype,
                device_map=device_map,
                load_4bit=load_4bit,
                local_files_only=local_files_only,
            )
            loaded_name = candidate
            break
        except Exception as exc:
            warn(f"Failed to load {candidate}: {exc}")
    rows = []
    if tokenizer is None or model is None:
        warn(f"No probability model could be loaded for {load_name}; writing NaN features.")
        for _, row in df.iterrows():
            feats = empty_feature_row(prefix)
            feats["id"] = row["id"]
            rows.append(feats)
    else:
        if loaded_name != load_name:
            warn(f"Using fallback model {loaded_name} for requested {load_name}; output prefix remains {prefix}.")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Probability {prefix}"):
            try:
                feats = token_loss_features(row["text"], tokenizer, model, max_length=max_length, prefix=prefix)
            except Exception as exc:
                warn(f"Probability feature failed for id={row['id']}: {exc}")
                feats = empty_feature_row(prefix)
            feats["id"] = row["id"]
            rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    del tokenizer, model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_key", choices=list(config.MODEL_REGISTRY), default=None)
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--output", default="")
    parser.add_argument("--dtype", default=config.DTYPE, choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--allow_fallback", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--auto_download", action="store_true")
    parser.add_argument("--model_root", default=None)
    args = parser.parse_args()
    config.ensure_dirs()
    model_for_name = args.model_key or args.model or config.DEFAULT_SMALL_MODEL
    if args.model_key:
        prefix = config.MODEL_KEY_PREFIX[args.model_key]
    else:
        prefix = model_safe_name(model_for_name)
    output = args.output or str(config.FEATURE_DIR / f"probability_{prefix}.csv")
    build_probability_features(
        args.input,
        output,
        args.model,
        args.dtype,
        args.max_length,
        args.device_map,
        args.load_4bit,
        args.allow_fallback,
        model_key=args.model_key,
        local_files_only=args.local_files_only,
        auto_download=args.auto_download,
        model_root=args.model_root,
    )


if __name__ == "__main__":
    main()
