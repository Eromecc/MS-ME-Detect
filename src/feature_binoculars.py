"""Simplified Binoculars-inspired dual-model probability contrast features."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import config
    from .feature_probability import load_causal_lm, token_loss_features
    from .preprocess import load_dataset
    from .utils import model_safe_name, safe_divide, warn, write_csv
except ImportError:
    import config
    from feature_probability import load_causal_lm, token_loss_features
    from preprocess import load_dataset
    from utils import model_safe_name, safe_divide, warn, write_csv


def pair_prefix(observer_model: str, performer_model: str) -> str:
    def short(m: str) -> str:
        s = model_safe_name(m).replace("qwen25_", "")
        return s.replace("_", "_")

    return f"bino_{short(observer_model)}_{short(performer_model)}"


def resolve_model(model: str | None, model_key: str | None, local_files_only: bool = False) -> tuple[str, str]:
    """Resolve a Binoculars model argument to a path/repo and stable label."""
    if model_key:
        local_dir = config.get_model_local_path(model_key)
        if not config.is_local_model_ready(local_dir):
            raise FileNotFoundError(
                "Local model not found or incomplete. "
                "Run: python scripts/download_models.py --models small medium large xl"
            )
        return local_dir, config.MODEL_KEY_PREFIX.get(model_key, model_key)
    requested = model or config.DEFAULT_SMALL_MODEL
    resolved = config.resolve_model_path(requested, online=not local_files_only)
    return resolved, model_safe_name(requested)


def empty_pair(prefix: str) -> dict[str, float]:
    names = [
        "ppl_observer",
        "ppl_performer",
        "loss_mean_observer",
        "loss_mean_performer",
        "loss_std_observer",
        "loss_std_performer",
        "ppl_ratio",
        "loss_mean_diff",
        "loss_std_diff",
        "binoculars_style_score",
    ]
    return {f"{prefix}_{n}": np.nan for n in names}


def build_pair_features(
    df: pd.DataFrame,
    observer_model: str,
    performer_model: str,
    dtype: str,
    max_length: int,
    device_map: str | None,
    load_4bit: bool,
    observer_label: str | None = None,
    performer_label: str | None = None,
    local_files_only: bool = False,
    allow_missing_models: bool = False,
):
    observer_label = observer_label or model_safe_name(observer_model)
    performer_label = performer_label or model_safe_name(performer_model)
    prefix = f"bino_{observer_label.replace('qwen25_', '')}_{performer_label.replace('qwen25_', '')}"
    try:
        obs_tok, obs_model = load_causal_lm(observer_model, dtype=dtype, device_map=device_map, load_4bit=load_4bit, local_files_only=local_files_only)
        perf_tok, perf_model = load_causal_lm(performer_model, dtype=dtype, device_map=device_map, load_4bit=load_4bit, local_files_only=local_files_only)
    except Exception as exc:
        message = f"Binoculars-style pair load failed ({observer_model}, {performer_model}): {exc}"
        if not allow_missing_models:
            raise RuntimeError(message) from exc
        warn(message)
        return pd.DataFrame([{"id": row["id"], **empty_pair(prefix)} for _, row in df.iterrows()])
    rows = []
    op = observer_label
    pp = performer_label
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Binoculars {prefix}"):
        try:
            o = token_loss_features(row["text"], obs_tok, obs_model, max_length=max_length, prefix=op)
            p = token_loss_features(row["text"], perf_tok, perf_model, max_length=max_length, prefix=pp)
            oppl, pppl = o[f"{op}_ppl"], p[f"{pp}_ppl"]
            olm, plm = o[f"{op}_loss_mean"], p[f"{pp}_loss_mean"]
            osd, psd = o[f"{op}_loss_std"], p[f"{pp}_loss_std"]
            feats = {
                f"{prefix}_ppl_observer": oppl,
                f"{prefix}_ppl_performer": pppl,
                f"{prefix}_loss_mean_observer": olm,
                f"{prefix}_loss_mean_performer": plm,
                f"{prefix}_loss_std_observer": osd,
                f"{prefix}_loss_std_performer": psd,
                f"{prefix}_ppl_ratio": safe_divide(oppl, pppl),
                f"{prefix}_loss_mean_diff": olm - plm,
                f"{prefix}_loss_std_diff": osd - psd,
                f"{prefix}_binoculars_style_score": safe_divide(olm, plm),
            }
        except Exception as exc:
            warn(f"Binoculars-style feature failed for id={row['id']}: {exc}")
            feats = empty_pair(prefix)
        feats["id"] = row["id"]
        rows.append(feats)
    del obs_tok, obs_model, perf_tok, perf_model
    gc.collect()
    return pd.DataFrame(rows)


def build_features(
    input_path: str | Path,
    output_path: str | Path,
    observer_model: str | None = None,
    performer_model: str | None = None,
    dtype: str = "bfloat16",
    max_length: int = 1024,
    device_map: str | None = None,
    load_4bit: bool = False,
    include_large_pair: bool = False,
    observer_key: str | None = None,
    performer_key: str | None = None,
    observer_key_2: str | None = None,
    performer_key_2: str | None = None,
    local_files_only: bool = False,
    allow_missing_models: bool = False,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    if observer_model is None and observer_key is None:
        observer_key = "small"
    if performer_model is None and performer_key is None:
        performer_key = "medium"
    obs, obs_label = resolve_model(observer_model, observer_key, local_files_only=local_files_only)
    perf, perf_label = resolve_model(performer_model, performer_key, local_files_only=local_files_only)
    frames = [
        build_pair_features(
            df,
            obs,
            perf,
            dtype,
            max_length,
            device_map,
            load_4bit,
            obs_label,
            perf_label,
            local_files_only,
            allow_missing_models,
        )
    ]
    if include_large_pair:
        obs2, obs2_label = resolve_model(None, observer_key_2 or "large", local_files_only=local_files_only)
        perf2, perf2_label = resolve_model(None, performer_key_2 or "xl", local_files_only=local_files_only)
        frames.append(
            build_pair_features(
                df,
                obs2,
                perf2,
                dtype,
                max_length,
                device_map,
                load_4bit,
                obs2_label,
                perf2_label,
                local_files_only,
                allow_missing_models,
            )
        )
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="id", how="outer")
    write_csv(out, output_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--output", default=str(config.FEATURE_DIR / "binoculars_features.csv"))
    parser.add_argument("--observer_model", default=None)
    parser.add_argument("--performer_model", default=None)
    parser.add_argument("--observer_key", choices=list(config.MODEL_REGISTRY), default=None)
    parser.add_argument("--performer_key", choices=list(config.MODEL_REGISTRY), default=None)
    parser.add_argument("--observer_key_2", choices=list(config.MODEL_REGISTRY), default="large")
    parser.add_argument("--performer_key_2", choices=list(config.MODEL_REGISTRY), default="xl")
    parser.add_argument("--dtype", default=config.DTYPE)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--include_large_pair", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_missing_models", action="store_true")
    args = parser.parse_args()
    config.ensure_dirs()
    build_features(
        args.input,
        args.output,
        args.observer_model,
        args.performer_model,
        args.dtype,
        args.max_length,
        args.device_map,
        args.load_4bit,
        args.include_large_pair,
        args.observer_key,
        args.performer_key,
        args.observer_key_2,
        args.performer_key_2,
        args.local_files_only,
        args.allow_missing_models,
    )


if __name__ == "__main__":
    main()
