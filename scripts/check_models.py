#!/usr/bin/env python
"""Check whether local Qwen2.5 model directories are ready."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024**3)


def tokenizer_ready(path: Path) -> bool:
    return any((path / name).exists() for name in ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json"])


def inspect_model(model_key: str, model_root: str | None = None) -> dict:
    entry = dict(config.get_model_entry(model_key))
    if model_root:
        entry["local_dir"] = str(Path(model_root) / Path(entry["local_dir"]).name)
    local_dir = Path(entry["local_dir"])
    missing = []
    if not local_dir.exists():
        missing.append("directory")
    if not (local_dir / "config.json").exists():
        missing.append("config.json")
    if not tokenizer_ready(local_dir):
        missing.append("tokenizer files")
    if not list(local_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    status = "READY" if not missing else ("MISSING" if "directory" in missing else "INCOMPLETE")
    return {
        "key": model_key,
        "repo_id": entry["repo_id"],
        "local_dir": str(local_dir),
        "status": status,
        "total_size_gb": round(size_gb(local_dir), 3),
        "missing_files": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Qwen2.5 model readiness.")
    parser.add_argument("--models", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    records = [inspect_model(model_key, args.model_root) for model_key in args.models]
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        for rec in records:
            missing = ", ".join(rec["missing_files"]) if rec["missing_files"] else "-"
            print(f"{rec['key']:14s} {rec['status']:10s} {rec['total_size_gb']:8.3f} GB  {rec['local_dir']}  missing: {missing}")
    return 0 if all(rec["status"] == "READY" for rec in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

