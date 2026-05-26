#!/usr/bin/env python3
"""Build head/tail frozen encoder embedding features with multi-seed projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META = [
    "id",
    "text",
    "label",
    "source_dataset",
    "language",
    "domain",
    "generator",
    "source",
    "split",
    "extra_metadata",
    "text_hash",
    "attack_type",
    "type",
    "topic",
]
CANONICAL = {
    "train": ROOT / "data/reproduction_datasets/fakespot_like_train.csv",
    "val": ROOT / "data/reproduction_datasets/fakespot_like_val.csv",
    "all_samples": ROOT / "data/test/all_samples_prepared.csv",
}
MODEL_SPECS = [
    "deberta=/vepfs-mlp2/queue010/20252203113/models/microsoft__deberta-v3-large",
    "roberta=/vepfs-mlp2/queue010/20252203113/models/roberta-large",
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_named(items: Iterable[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name.strip()] = resolve(path.strip())
    return out


def load_model(path: Path, dtype_name: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok_kwargs = {"local_files_only": True, "trust_remote_code": False}
    if "deberta-v3" in str(path):
        tok_kwargs["fix_mistral_regex"] = True
    tok = AutoTokenizer.from_pretrained(path, **tok_kwargs)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False, torch_dtype=dtype)
    if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    model.eval()
    return tok, model


def projection(hidden_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + hidden_dim * 13 + out_dim)
    return rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(hidden_dim, out_dim)).astype("float32")


def pooled_batch(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    truncation_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = truncation_side
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tokenizer.truncation_side = old_side
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask = enc.get("attention_mask")
    if mask is None:
        mask = torch.ones(out.shape[:2], dtype=torch.long, device=out.device)
    mask_f = mask.unsqueeze(-1).to(out.dtype)
    mean = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    cls = out[:, 0, :]
    return cls.float().cpu().numpy(), mean.float().cpu().numpy()


def pool_feature_frame(
    prefix: str,
    cls: np.ndarray,
    mean: np.ndarray,
    proj_dim: int,
    seeds: list[int],
) -> pd.DataFrame:
    cls_norm = np.linalg.norm(cls, axis=1)
    mean_norm = np.linalg.norm(mean, axis=1)
    data = {
        f"{prefix}_cls_norm": cls_norm.astype("float32"),
        f"{prefix}_mean_norm": mean_norm.astype("float32"),
        f"{prefix}_cls_mean_cos": (
            np.einsum("ij,ij->i", cls, mean) / np.maximum(cls_norm * mean_norm, 1e-12)
        ).astype("float32"),
    }
    frames = [pd.DataFrame(data)]
    for seed in seeds:
        pc = projection(cls.shape[1], proj_dim, seed + 17)
        pm = projection(mean.shape[1], proj_dim, seed + 31)
        cls_p = cls @ pc
        mean_p = mean @ pm
        frames.append(pd.DataFrame(cls_p, columns=[f"{prefix}_s{seed}_cls_rp{j:03d}" for j in range(proj_dim)]))
        frames.append(pd.DataFrame(mean_p, columns=[f"{prefix}_s{seed}_mean_rp{j:03d}" for j in range(proj_dim)]))
    return pd.concat(frames, axis=1)


def build_model_split(
    df: pd.DataFrame,
    name: str,
    path: Path,
    batch_size: int,
    max_length: int,
    proj_dim: int,
    seeds: list[int],
    dtype: str,
    device: str,
) -> pd.DataFrame:
    tok, model = load_model(path, dtype, device)
    frames = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        texts = batch["text"].astype(str).tolist()
        head_cls, head_mean = pooled_batch(texts, tok, model, device, max_length, "right")
        tail_cls, tail_mean = pooled_batch(texts, tok, model, device, max_length, "left")
        head_tail_denom = np.maximum(np.linalg.norm(head_mean, axis=1) * np.linalg.norm(tail_mean, axis=1), 1e-12)
        batch_frame = pd.concat(
            [
                pool_feature_frame(f"embseg_{name}_head", head_cls, head_mean, proj_dim, seeds),
                pool_feature_frame(f"embseg_{name}_tail", tail_cls, tail_mean, proj_dim, seeds),
                pd.DataFrame(
                    {
                        f"embseg_{name}_head_tail_mean_cos": (
                            np.einsum("ij,ij->i", head_mean, tail_mean) / head_tail_denom
                        ).astype("float32"),
                        f"embseg_{name}_head_tail_mean_l2": np.linalg.norm(head_mean - tail_mean, axis=1).astype(
                            "float32"
                        ),
                    }
                ),
            ],
            axis=1,
        )
        frames.append(batch_frame)
        print(f"{name} processed {min(start + len(batch), len(df))}/{len(df)}", flush=True)
    del tok, model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return pd.concat(frames, axis=0, ignore_index=True)


def build_split(split: str, output_root: Path, models: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    meta = pd.read_csv(CANONICAL[split])
    out = meta[[c for c in META if c in meta.columns]].copy()
    seeds = [int(x) for x in args.seeds]
    for name, path in models.items():
        part = build_model_split(
            meta,
            name,
            path,
            int(args.batch_size),
            int(args.max_length),
            int(args.proj_dim),
            seeds,
            str(args.dtype),
            str(args.device),
        )
        out = pd.concat([out.reset_index(drop=True), part.reset_index(drop=True)], axis=1)
    target = output_root / "fakespot_like" / split / "all_features.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return {"split": split, "rows": int(len(out)), "features": int(len(out.columns) - len([c for c in out.columns if c in META])), "path": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_SPECS)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "all_samples"])
    parser.add_argument("--output_dir", default="features_embedding_segment_v1")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260525, 20260526, 20260527])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    models = parse_named(args.models)
    output_root = resolve(args.output_dir)
    manifest = {"models": {k: str(v) for k, v in models.items()}, "splits": [], "args": vars(args)}
    for split in args.splits:
        manifest["splits"].append(build_split(split, output_root, models, args))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "embedding_segment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
