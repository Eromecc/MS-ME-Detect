#!/usr/bin/env python3
"""Build frozen LM embedding-pool features with deterministic random projection."""

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
    dtype = torch.float32
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tok, model


def projection(hidden_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + hidden_dim * 13 + out_dim)
    mat = rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(hidden_dim, out_dim))
    return mat.astype("float32")


def pooled_batch(texts: list[str], tokenizer: Any, model: Any, device: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
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


def build_model_split(
    df: pd.DataFrame,
    name: str,
    path: Path,
    batch_size: int,
    max_length: int,
    proj_dim: int,
    seed: int,
    dtype: str,
    device: str,
) -> pd.DataFrame:
    tok, model = load_model(path, dtype, device)
    rows = []
    proj_cls = proj_mean = None
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        cls, mean = pooled_batch(batch["text"].astype(str).tolist(), tok, model, device, max_length)
        if proj_cls is None or proj_mean is None:
            proj_cls = projection(cls.shape[1], proj_dim, seed + 17)
            proj_mean = projection(mean.shape[1], proj_dim, seed + 31)
        cls_p = cls @ proj_cls
        mean_p = mean @ proj_mean
        for i in range(len(batch)):
            item = {
                f"emb_{name}_cls_norm": float(np.linalg.norm(cls[i])),
                f"emb_{name}_mean_norm": float(np.linalg.norm(mean[i])),
                f"emb_{name}_cls_mean_cos": float(np.dot(cls[i], mean[i]) / max(np.linalg.norm(cls[i]) * np.linalg.norm(mean[i]), 1e-12)),
            }
            for j, value in enumerate(cls_p[i]):
                item[f"emb_{name}_cls_rp{j:03d}"] = float(value)
            for j, value in enumerate(mean_p[i]):
                item[f"emb_{name}_mean_rp{j:03d}"] = float(value)
            rows.append(item)
    del tok, model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return pd.DataFrame(rows)


def build_split(split: str, output_root: Path, models: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    meta = pd.read_csv(CANONICAL[split])
    out = meta[[c for c in META if c in meta.columns]].copy()
    for name, path in models.items():
        part = build_model_split(
            meta,
            name,
            path,
            int(args.batch_size),
            int(args.max_length),
            int(args.proj_dim),
            int(args.seed),
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
    parser.add_argument("--output_dir", default="features_embedding_pool_v1")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    models = parse_named(args.models)
    output_root = resolve(args.output_dir)
    manifest = {"models": {k: str(v) for k, v in models.items()}, "splits": [], "args": vars(args)}
    for split in args.splits:
        manifest["splits"].append(build_split(split, output_root, models, args))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "embedding_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
