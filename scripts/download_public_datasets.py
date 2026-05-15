#!/usr/bin/env python3
"""Download public English LLM text detection datasets and rebuild the unified CSV."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


GHOSTBUSTER_REPO = "https://github.com/vivek3141/ghostbuster-data.git"
M4_REPO = "https://github.com/mbzuai-nlp/M4.git"


def run_command(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired as exc:
        return False, f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
    return completed.returncode == 0, output


def ensure_clean_target(path: Path, force: bool) -> tuple[bool, str]:
    if path.exists():
        has_content = path.is_file() or any(path.iterdir()) if path.is_dir() else True
        if has_content and not force:
            return False, "exists_non_empty"
        if force:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return True, "ready"


def clone_repo(url: str, target: Path, force: bool) -> tuple[str, str | None]:
    should_continue, state = ensure_clean_target(target, force)
    if not should_continue and state == "exists_non_empty":
        return "skipped", None
    ok, output = run_command(["git", "clone", "--depth", "1", url, str(target)], timeout=45)
    return ("success" if ok else "failed"), (None if ok else output)


def collect_candidate_files(root: Path, limit: int = 20) -> tuple[int, list[str]]:
    exts = {".csv", ".json", ".jsonl", ".parquet"}
    paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts]
    rels = [str(path.relative_to(root)) for path in sorted(paths)[:limit]]
    return len(paths), rels


def download_ghostbuster(output_root: Path, force: bool) -> dict:
    target = output_root / "ghostbuster"
    status, error = clone_repo(GHOSTBUSTER_REPO, target, force)
    result = {
        "dataset": "ghostbuster",
        "status": status,
        "local_path": str(target),
        "error": error,
        "sampled_only": False,
    }
    if status == "success":
        data_dir = target / "data"
        if data_dir.exists():
            print(f"[ghostbuster] Downloaded successfully: {target}")
            print(f"[ghostbuster] Verified path exists: {data_dir}")
        else:
            warning = f"Expected data directory missing: {data_dir}"
            print(f"[ghostbuster] Warning: {warning}")
            result["warning"] = warning
    elif status == "skipped":
        print(f"[ghostbuster] Skipping existing non-empty directory: {target}")
    else:
        print(f"[ghostbuster] Download failed: {error}")
    return result


def download_m4(output_root: Path, force: bool) -> dict:
    target = output_root / "m4"
    status, error = clone_repo(M4_REPO, target, force)
    result = {
        "dataset": "m4",
        "status": status,
        "local_path": str(target),
        "error": error,
        "sampled_only": False,
    }
    if status == "success":
        count, sample_paths = collect_candidate_files(target, limit=20)
        print(f"[m4] Downloaded successfully: {target}")
        print(f"[m4] Candidate csv/json/jsonl/parquet files found: {count}")
        for rel_path in sample_paths:
            print(f"  - {rel_path}")
        result["candidate_file_count"] = count
        result["candidate_file_examples"] = sample_paths
    elif status == "skipped":
        count, sample_paths = collect_candidate_files(target, limit=20)
        print(f"[m4] Skipping existing non-empty directory: {target}")
        print(f"[m4] Candidate csv/json/jsonl/parquet files found: {count}")
        for rel_path in sample_paths:
            print(f"  - {rel_path}")
        result["candidate_file_count"] = count
        result["candidate_file_examples"] = sample_paths
    else:
        print(f"[m4] Download failed: {error}")
    return result


def download_raid_sample(output_root: Path, max_raid_samples: int, seed: int, no_raid_full_download: bool) -> dict:
    target_dir = output_root / "raid"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "raid_sample.jsonl"
    result = {
        "dataset": "raid",
        "status": "failed",
        "local_path": str(target_file),
        "error": None,
        "sampled_only": True,
        "max_raid_samples": int(max_raid_samples),
        "no_raid_full_download": bool(no_raid_full_download),
    }

    if not no_raid_full_download:
        msg = "RAID full download is disabled by policy; rerun with --no_raid_full_download."
        print(f"[raid] {msg}")
        result["status"] = "skipped"
        result["error"] = msg
        return result

    try:
        from datasets import load_dataset
    except Exception as exc:
        msg = f"'datasets' is not installed. Install it with: pip install datasets. Original error: {exc}"
        print(f"[raid] {msg}")
        result["status"] = "failed"
        result["error"] = msg
        return result

    try:
        stream = load_dataset("liamdugan/raid", split="train", streaming=True)
    except Exception as exc:
        msg = f"Streaming load failed. Network, permission, or Hugging Face access may be unavailable. Error: {exc}"
        print(f"[raid] {msg}")
        result["status"] = "failed"
        result["error"] = msg
        return result

    rows = []
    try:
        for idx, item in enumerate(stream.shuffle(seed=seed, buffer_size=min(max_raid_samples * 2, 10000))):
            if idx >= max_raid_samples:
                break
            rows.append(item)
    except Exception as exc:
        msg = f"Streaming iteration failed. Network access may be unavailable or the dataset may require authorization. Error: {exc}"
        print(f"[raid] {msg}")
        result["status"] = "failed"
        result["error"] = msg
        return result

    if not rows:
        msg = "No rows were returned from RAID streaming."
        print(f"[raid] {msg}")
        result["status"] = "failed"
        result["error"] = msg
        return result

    with target_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[raid] Saved streaming sample to: {target_file}")
    print(f"[raid] Sample rows written: {len(rows)}")
    result["status"] = "success"
    result["sample_count"] = len(rows)
    return result


def write_manifest(path: Path, requested_dataset: str, results: list[dict], max_raid_samples: int) -> None:
    manifest = {
        "dataset_requested": requested_dataset,
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "max_raid_samples": int(max_raid_samples),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_prepare_script(repo_root: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_english_public_datasets.py"),
        "--dataset",
        "all",
        "--input_root",
        "data/raw",
        "--output",
        "data/dataset_english_v1.csv",
        "--max_per_group",
        "1000",
        "--seed",
        "42",
    ]
    ok, output = run_command(cmd, cwd=repo_root)
    return ok, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public English datasets and rebuild the unified dataset CSV.")
    parser.add_argument("--dataset", choices=["ghostbuster", "m4", "raid", "all"], default="ghostbuster")
    parser.add_argument("--output_root", default="data/raw")
    parser.add_argument("--max_raid_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_raid_full_download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = ["ghostbuster", "m4", "raid"] if args.dataset == "all" else [args.dataset]
    results = []

    for dataset_name in selected:
        if dataset_name == "ghostbuster":
            results.append(download_ghostbuster(output_root, args.force))
        elif dataset_name == "m4":
            results.append(download_m4(output_root, args.force))
        elif dataset_name == "raid":
            results.append(download_raid_sample(output_root, args.max_raid_samples, args.seed, args.no_raid_full_download))

    manifest_path = output_root / "download_manifest.json"
    write_manifest(manifest_path, args.dataset, results, args.max_raid_samples)
    print(f"Saved download manifest: {manifest_path}")

    ok, output = run_prepare_script(repo_root)
    if output:
        print(output)
    if not ok:
        print("Warning: prepare_english_public_datasets.py returned a non-zero status, but downloads were still recorded.")


if __name__ == "__main__":
    main()
