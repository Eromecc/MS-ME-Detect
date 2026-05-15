"""Qwen2.5 Base model probability and token-level loss features."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
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


def token_loss_sequence(text: str, tokenizer, model, max_length: int = 1024) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not str(text).strip():
        return np.asarray([], dtype=float)
    device = next(model.parameters()).device
    encoded = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if input_ids.shape[1] < 2:
        return np.asarray([], dtype=float)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    return losses.detach().float().cpu().numpy()


def token_loss_features_from_values(vals: np.ndarray, prefix: str = "model") -> dict[str, float]:
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


def token_loss_features(text: str, tokenizer, model, max_length: int = 1024, prefix: str = "model") -> dict[str, float]:
    return token_loss_features_from_values(token_loss_sequence(text, tokenizer, model, max_length=max_length), prefix=prefix)


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def token_loss_cache_path(output_dir: str | Path, model_name: str, dataset_name: str) -> Path:
    return Path(output_dir) / model_name / f"{dataset_name}_token_loss.jsonl.gz"


def read_cached_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    ids.add(str(item.get("id")))
                except Exception:
                    continue
    except Exception:
        return ids
    return ids


def write_token_loss_manifest(
    manifest_path: Path,
    *,
    model_name: str,
    dataset_name: str,
    cache_path: Path,
    token_counts: list[int],
    max_length: int,
    skipped_ids: list[str],
    failed_ids: list[str],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "cache_path": str(cache_path),
        "n_rows": int(len(token_counts)),
        "mean_token_count": float(np.mean(token_counts)) if token_counts else 0.0,
        "max_token_count": int(max(token_counts)) if token_counts else 0,
        "min_token_count": int(min(token_counts)) if token_counts else 0,
        "max_length": int(max_length),
        "loss_sequence_length_definition": "next-token cross-entropy losses; length is token_count=input_tokens_after_shift, usually tokenizer_length-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skipped_ids": skipped_ids,
        "failed_ids": failed_ids,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    save_token_loss: bool = False,
    token_loss_output_dir: str | Path = "features_token_loss",
    dataset_name: str | None = None,
    token_loss_model_name: str | None = None,
    resume: bool = False,
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
    cache_handle = None
    cache_path = None
    cached_ids: set[str] = set()
    token_counts: list[int] = []
    skipped_ids: list[str] = []
    failed_ids: list[str] = []
    dataset_name = dataset_name or Path(input_path).stem
    cache_model_name = token_loss_model_name or prefix
    if save_token_loss:
        cache_path = token_loss_cache_path(token_loss_output_dir, cache_model_name, dataset_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            cached_ids = read_cached_ids(cache_path)
        cache_handle = gzip.open(cache_path, "at", encoding="utf-8")
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
                losses = token_loss_sequence(row["text"], tokenizer, model, max_length=max_length)
                feats = token_loss_features_from_values(losses, prefix=prefix)
                if cache_handle is not None and str(row["id"]) not in cached_ids:
                    item = {
                        "id": row["id"],
                        "model_name": cache_model_name,
                        "token_count": int(losses.size),
                        "loss_sequence": [float(x) for x in losses.tolist()],
                        "rank_sequence": pd.Series(-losses).rank(method="average", pct=True).round(8).tolist() if losses.size else [],
                        "prob_sequence": [float(math.exp(-min(float(x), 50.0))) for x in losses.tolist()],
                        "text_hash": text_hash(row["text"]),
                        "max_length": int(max_length),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    cache_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    token_counts.append(int(losses.size))
                elif cache_handle is not None:
                    skipped_ids.append(str(row["id"]))
            except Exception as exc:
                warn(f"Probability feature failed for id={row['id']}: {exc}")
                feats = empty_feature_row(prefix)
                failed_ids.append(str(row["id"]))
            feats["id"] = row["id"]
            rows.append(feats)
    if cache_handle is not None:
        cache_handle.close()
        write_token_loss_manifest(
            cache_path.with_name(cache_path.name.replace(".jsonl.gz", "_manifest.json")),
            model_name=cache_model_name,
            dataset_name=dataset_name,
            cache_path=cache_path,
            token_counts=token_counts,
            max_length=max_length,
            skipped_ids=skipped_ids,
            failed_ids=failed_ids,
        )
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
    parser.add_argument("--save_token_loss", action="store_true")
    parser.add_argument("--token_loss_output_dir", default="features_token_loss")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--token_loss_model_name", default=None)
    parser.add_argument("--resume", action="store_true")
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
        save_token_loss=args.save_token_loss,
        token_loss_output_dir=args.token_loss_output_dir,
        dataset_name=args.dataset_name,
        token_loss_model_name=args.token_loss_model_name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
