#!/usr/bin/env python
"""Download Qwen2.5 models into the configured local model directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def model_entry_with_root(model_key: str, model_root: str) -> dict:
    entry = dict(config.get_model_entry(model_key))
    entry["local_dir"] = str(Path(model_root) / Path(entry["local_dir"]).name)
    return entry


def tokenizer_exists(local_dir: Path) -> bool:
    names = ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json"]
    return any((local_dir / name).exists() for name in names)


def directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024**3)


def verify_local_model(model_key: str, entry: dict) -> dict:
    local_dir = Path(entry["local_dir"])
    safetensors = list(local_dir.glob("*.safetensors")) if local_dir.exists() else []
    return {
        "model_key": model_key,
        "repo_id": entry["repo_id"],
        "local_dir": str(local_dir),
        "exists": local_dir.exists(),
        "config_exists": (local_dir / "config.json").exists(),
        "tokenizer_exists": tokenizer_exists(local_dir),
        "safetensors_count": len(safetensors),
        "total_size_gb": round(directory_size_gb(local_dir), 3),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def is_ready(record: dict) -> bool:
    return bool(record["exists"] and record["config_exists"] and record["tokenizer_exists"] and record["safetensors_count"] > 0)


def ensure_backend(backend: str) -> str | None:
    command = "hf" if backend == "hf" else "modelscope"
    path = shutil.which(command)
    if path:
        return path
    if backend == "hf":
        print("Missing 'hf' command. Install it with: pip install -U huggingface_hub")
    else:
        print("Missing 'modelscope' command. Install it with: pip install -U modelscope")
    return None


def download_command(backend: str, repo_id: str, local_dir: str) -> list[str]:
    if backend == "hf":
        return ["hf", "download", repo_id, "--local-dir", local_dir]
    return ["modelscope", "download", "--model", repo_id, "--local_dir", local_dir]


def save_manifest(records: list[dict], model_root: str) -> None:
    path = Path(model_root) / "model_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {path}")


def prepare_env(model_root: str) -> dict:
    env = os.environ.copy()
    cache = "/vepfs-mlp2/queue010/20252203113/hf_cache"
    env.setdefault("HF_HOME", cache)
    env.setdefault("TRANSFORMERS_CACHE", cache)
    env.setdefault("MODEL_ROOT", model_root)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print(f"HF_HOME={env['HF_HOME']}")
    print(f"TRANSFORMERS_CACHE={env['TRANSFORMERS_CACHE']}")
    print(f"MODEL_ROOT={env['MODEL_ROOT']}")
    return env


def selected_models(models: list[str], include_instruct: bool) -> list[str]:
    out = list(dict.fromkeys(models))
    if include_instruct and "instruct_large" not in out:
        out.append("instruct_large")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen2.5 models to local disk.")
    parser.add_argument("--models", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--model_root", default=config.MODEL_ROOT)
    parser.add_argument("--backend", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include_instruct", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    env = prepare_env(args.model_root)
    backend_cmd = ensure_backend(args.backend)
    if backend_cmd is None and not args.dry_run:
        return 1

    records = []
    for model_key in selected_models(args.models, args.include_instruct):
        entry = model_entry_with_root(model_key, args.model_root)
        local_dir = Path(entry["local_dir"])
        before = verify_local_model(model_key, entry)
        if args.skip_existing and not args.force and is_ready(before):
            print(f"READY, skipping {model_key}: {local_dir}")
            records.append(before)
            continue
        if args.force and local_dir.exists():
            print(f"--force set; removing existing directory before download: {local_dir}")
            if not args.dry_run:
                shutil.rmtree(local_dir)
        cmd = download_command(args.backend, entry["repo_id"], str(local_dir))
        print(" ".join(cmd))
        if not args.dry_run:
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True, env=env)
        record = verify_local_model(model_key, entry)
        if not is_ready(record):
            print(f"Warning: downloaded model is incomplete: {model_key}")
        records.append(record)

    save_manifest(records, args.model_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

