"""Hidden-state trajectory cache for strict Text-Koopman experiments.

The cache intentionally stores hidden-state tensors only. It does not store raw
token ids, token strings, or raw text.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def safe_model_name(model_name: str) -> str:
    return str(model_name).replace("/", "_").replace("-", "_").replace(".", "_")


def hidden_state_dataset_dir(root: str | Path, model_name: str, dataset_name: str) -> Path:
    return Path(root) / safe_model_name(model_name) / dataset_name


def manifest_path(root: str | Path, model_name: str, dataset_name: str) -> Path:
    return Path(root) / safe_model_name(model_name) / f"{dataset_name}_hidden_state_manifest.json"


def read_hidden_manifest(root: str | Path, model_name: str, dataset_name: str) -> dict:
    path = manifest_path(root, model_name, dataset_name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cached_hidden_ids(root: str | Path, model_name: str, dataset_name: str, max_length: int | None = None) -> set[str]:
    d = hidden_state_dataset_dir(root, model_name, dataset_name)
    ids: set[str] = set()
    for path in d.glob("shard_*.pt"):
        try:
            import torch

            payload = torch.load(path, map_location="cpu")
            for record in payload.get("records", []):
                if max_length is not None and int(record.get("max_length", -1)) != int(max_length):
                    continue
                ids.add(str(record["id"]))
        except Exception:
            continue
    if ids:
        return ids
    manifest = read_hidden_manifest(root, model_name, dataset_name)
    if manifest.get("ids") and (max_length is None or int(manifest.get("max_length", -1)) == int(max_length)):
        return set(map(str, manifest["ids"]))
    return set()


def iter_hidden_records(root: str | Path, model_name: str, dataset_name: str):
    """Yield cached records with keys id, hidden_states, seq_len, metadata."""
    import torch

    d = hidden_state_dataset_dir(root, model_name, dataset_name)
    for path in sorted(d.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu")
        for record in payload.get("records", []):
            yield record


def rebuild_hidden_manifest_from_shards(root: str | Path, model_name: str, dataset_name: str) -> dict:
    """Rebuild a manifest from shard contents.

    This is useful when an older smoke-run manifest is stale but full shard
    files already exist. It reads only hidden-state metadata and never raw text
    or token ids.
    """
    ids: list[str] = []
    seq_lens: list[int] = []
    hidden_size = None
    max_length = None
    dtype = None
    for record in iter_hidden_records(root, model_name, dataset_name):
        ids.append(str(record["id"]))
        seq_lens.append(int(record.get("seq_len", record["hidden_states"].shape[0])))
        hidden_size = hidden_size or int(record.get("hidden_size", record["hidden_states"].shape[1]))
        max_length = max_length or int(record.get("max_length", 0))
        dtype = dtype or str(record["hidden_states"].dtype).replace("torch.", "")
    payload = {
        "created_at": now(),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_rows": int(len(set(ids))),
        "failed_ids": [],
        "skipped_ids": [],
        "mean_seq_len": float(np.mean(seq_lens)) if seq_lens else None,
        "max_seq_len": int(np.max(seq_lens)) if seq_lens else None,
        "min_seq_len": int(np.min(seq_lens)) if seq_lens else None,
        "hidden_size": hidden_size,
        "dtype": dtype,
        "max_length": max_length,
        "ids": sorted(set(ids)),
        "storage_note": "Stores last-layer hidden states only; no raw token ids, token strings, or raw text.",
        "rebuilt_from_shards": True,
    }
    _write_manifest(manifest_path(root, model_name, dataset_name), payload)
    return payload


def load_hidden_cache_map(root: str | Path, model_name: str, dataset_names: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for dataset_name in dataset_names:
        for record in iter_hidden_records(root, model_name, dataset_name):
            out[str(record["id"])] = record
    return out


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_hidden_state_cache(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    tokenizer,
    model,
    output_root: str | Path = "features_hidden_states",
    text_col: str = "text",
    id_col: str = "id",
    max_length: int = 256,
    batch_size: int = 4,
    shard_size: int = 128,
    resume: bool = True,
    dtype: str = "float16",
) -> dict:
    """Extract last-layer hidden states into sharded torch files.

    Hidden size is read from ``model.config.hidden_size`` and never hard-coded.
    """
    import torch
    from tqdm import tqdm

    out_dir = hidden_state_dataset_dir(output_root, model_name, dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = manifest_path(output_root, model_name, dataset_name)
    existing = cached_hidden_ids(output_root, model_name, dataset_name, max_length=max_length) if resume else set()
    hidden_size = int(getattr(model.config, "hidden_size"))
    device = next(model.parameters()).device
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    rows = df.copy()
    rows[id_col] = rows[id_col].astype(str)
    todo = rows[~rows[id_col].isin(existing)].copy()
    records: list[dict] = []
    failed_ids: list[str] = []
    skipped_ids: list[str] = sorted(existing.intersection(set(rows[id_col])))
    seq_lens: list[int] = []
    shard_idx = len(list(out_dir.glob("shard_*.pt"))) if resume else 0
    fallback_events: list[dict] = []

    def flush() -> None:
        nonlocal records, shard_idx
        if not records:
            return
        path = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({"records": records, "model_name": model_name, "dataset_name": dataset_name}, path)
        records = []
        shard_idx += 1

    def is_oom(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg or "cublas" in msg

    def current_device() -> torch.device:
        return next(model.parameters()).device

    def run_batch(batch: pd.DataFrame, *, allow_split: bool = True, allow_cpu: bool = True) -> None:
        nonlocal device
        try:
            encoded = tokenizer(
                batch[text_col].astype(str).tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            device = current_device()
            attention_mask = encoded["attention_mask"].to(device)
            model_inputs = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[-1].detach()
            for i, (_, row) in enumerate(batch.iterrows()):
                rid = str(row[id_col])
                valid_len = int(attention_mask[i].sum().item())
                h = hidden[i, :valid_len].to(dtype=torch_dtype).cpu()
                records.append(
                    {
                        "id": rid,
                        "text_hash": text_hash(row[text_col]),
                        "seq_len": valid_len,
                        "valid_len": valid_len,
                        "hidden_states": h,
                        "model_name": model_name,
                        "max_length": int(max_length),
                        "hidden_size": hidden_size,
                    }
                )
                seq_lens.append(valid_len)
                if len(records) >= shard_size:
                    flush()
        except RuntimeError as exc:
            if is_oom(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if allow_split and len(batch) > 1:
                    fallback_events.append({"type": "split_batch", "batch_size": int(len(batch)), "reason": str(exc)[:200]})
                    mid = len(batch) // 2
                    run_batch(batch.iloc[:mid], allow_split=True, allow_cpu=allow_cpu)
                    run_batch(batch.iloc[mid:], allow_split=True, allow_cpu=allow_cpu)
                    return
                if allow_cpu and current_device().type != "cpu":
                    fallback_events.append({"type": "cpu_fallback", "batch_size": int(len(batch)), "reason": str(exc)[:200]})
                    model.to("cpu")
                    device = torch.device("cpu")
                    run_batch(batch, allow_split=False, allow_cpu=False)
                    return
            failed_ids.extend(batch[id_col].astype(str).tolist())
        except Exception:
            failed_ids.extend(batch[id_col].astype(str).tolist())

    model.eval()
    for start in tqdm(range(0, len(todo), batch_size), desc=f"hidden:{dataset_name}"):
        batch = todo.iloc[start : start + batch_size]
        run_batch(batch)
    flush()

    all_ids = sorted(cached_hidden_ids(output_root, model_name, dataset_name))
    payload = {
        "created_at": now(),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_rows": int(len(all_ids)),
        "failed_ids": failed_ids,
        "skipped_ids": skipped_ids,
        "mean_seq_len": float(np.mean(seq_lens)) if seq_lens else None,
        "max_seq_len": int(np.max(seq_lens)) if seq_lens else None,
        "min_seq_len": int(np.min(seq_lens)) if seq_lens else None,
        "hidden_size": hidden_size,
        "dtype": dtype,
        "max_length": int(max_length),
        "ids": all_ids,
        "fallback_events": fallback_events,
        "storage_note": "Stores last-layer hidden states only; no raw token ids, token strings, or raw text.",
    }
    _write_manifest(man_path, payload)
    return payload
