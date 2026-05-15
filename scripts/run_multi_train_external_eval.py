#!/usr/bin/env python3
"""Orchestrate per-dataset training and external evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.feature_burstiness import build_features as build_burstiness
from src.feature_probability import build_probability_features
from src.feature_scale_response import build_scale_response_features
from src.feature_structure import build_features as build_structure
from src.merge_features import merge_features
from src.train_eval import train_and_evaluate
from src.utils import load_json, save_json

MODEL_ARG_TO_KEY = {
    "qwen25_1_5b": "small",
    "qwen25_7b": "medium",
    "qwen25_14b": "large",
    "qwen25_32b": "xl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-train / external-eval workflow.")
    parser.add_argument("--train_sets", nargs="+", default=["ghostbuster", "m4", "hc3_plus", "combined_public"])
    parser.add_argument("--external_test", default="data/test/all_samples.json")
    parser.add_argument("--run_features", action="store_true")
    parser.add_argument("--run_probability", action="store_true")
    parser.add_argument("--run_scale_response", action="store_true")
    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_external_eval", action="store_true")
    parser.add_argument("--feature_mode", choices=["basic", "allfeatures"], default="basic")
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument("--reuse_existing_features", action="store_true")
    parser.add_argument("--models", nargs="+", default=["qwen25_1_5b", "qwen25_7b", "qwen25_14b", "qwen25_32b"])
    parser.add_argument("--skip_missing_models", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_sample_csv(path: Path, max_rows: int | None, seed: int) -> Path:
    if max_rows is None:
        return path
    df = pd.read_csv(path)
    if len(df) <= max_rows:
        return path
    out_path = path.with_name(path.stem + f"_sample_{max_rows}.csv")
    df.sample(n=max_rows, random_state=seed).to_csv(out_path, index=False)
    return out_path


def has_both_classes(path: Path) -> bool:
    df = pd.read_csv(path)
    labels = sorted(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int).unique().tolist())
    if labels != [0, 1]:
        print(f"Warning: {path} has label set {labels}; will skip training.")
        return False
    return True


def dataset_stats(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    labels = pd.to_numeric(df["label"], errors="coerce")
    return {
        "rows": int(len(df)),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
    }


def csv_matches_ids(feature_csv: Path, data_csv: Path) -> bool:
    if not feature_csv.exists() or not data_csv.exists():
        return False
    try:
        left = pd.read_csv(feature_csv, usecols=["id"])
        right = pd.read_csv(data_csv, usecols=["id"])
    except Exception:
        return False
    if len(left) != len(right):
        return False
    return left["id"].astype(str).tolist() == right["id"].astype(str).tolist()


def reuse_or_copy(source_file: Path, target_file: Path, data_csv: Path) -> bool:
    if not source_file.exists():
        return False
    if not csv_matches_ids(source_file, data_csv):
        return False
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() and csv_matches_ids(target_file, data_csv):
        return True
    shutil.copy2(source_file, target_file)
    return True


def subset_feature_csv(source_file: Path, target_file: Path, data_csv: Path) -> bool:
    if not source_file.exists() or not data_csv.exists():
        return False
    try:
        source = pd.read_csv(source_file)
        base = pd.read_csv(data_csv, usecols=["id"])
    except Exception:
        return False
    if "id" not in source.columns:
        return False
    source["id"] = source["id"].astype(str)
    base["id"] = base["id"].astype(str)
    subset = base.merge(source, on="id", how="left")
    if len(subset) != len(base):
        return False
    target_file.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(target_file, index=False)
    return True


def ensure_external_prepared(root: Path, external_test: Path) -> Path:
    if external_test.suffix.lower() == ".csv":
        return external_test
    prepared = root / "data" / "test" / "all_samples_prepared.csv"
    if prepared.exists():
        return prepared
    cmd = [
        sys.executable,
        "scripts/prepare_external_test.py",
        "--input",
        str(external_test),
        "--output",
        str(prepared),
    ]
    subprocess.run(cmd, check=True, cwd=root)
    return prepared


def resolve_model_specs(models: list[str]) -> list[dict[str, str | bool]]:
    specs = []
    for model_name in models:
        model_key = MODEL_ARG_TO_KEY.get(model_name)
        if model_key is None:
            raise ValueError(f"Unsupported model alias: {model_name}")
        local_dir = config.get_model_local_path(model_key)
        ready = config.is_local_model_ready(local_dir)
        specs.append(
            {
                "model_name": model_name,
                "model_key": model_key,
                "local_dir": local_dir,
                "ready": ready,
            }
        )
    return specs


def model_device_map(device: str) -> str | None:
    if device == "auto":
        return "auto"
    return None


def feature_groups_used(feature_file: Path) -> list[str]:
    if not feature_file.exists():
        return []
    df = pd.read_csv(feature_file, nrows=1)
    cols = df.columns.tolist()
    groups = []
    if any(c.startswith("burst_") for c in cols):
        groups.append("burstiness")
    if any(c.startswith("struct_") for c in cols):
        groups.append("structure")
    if any(c.startswith("qwen25_") for c in cols):
        groups.append("probability")
    if any(c.startswith("scale_") for c in cols):
        groups.append("scale_response")
    return groups


def scale_response_scales(feature_dir: Path) -> list[str]:
    manifest = load_json(feature_dir / "scale_response_manifest.json", default={}) or {}
    return list(manifest.get("available_scales", []))


def build_basic_features(
    data_csv: Path,
    feature_dir: Path,
    *,
    reuse_existing_features: bool = False,
    basic_source_dir: Path | None = None,
) -> Path:
    feature_dir.mkdir(parents=True, exist_ok=True)
    burst_file = feature_dir / "burstiness_features.csv"
    struct_file = feature_dir / "structure_features.csv"

    if reuse_existing_features and basic_source_dir is not None:
        if reuse_or_copy(basic_source_dir / "burstiness_features.csv", burst_file, data_csv):
            print(f"Reused burstiness: {burst_file}")
        if reuse_or_copy(basic_source_dir / "structure_features.csv", struct_file, data_csv):
            print(f"Reused structure: {struct_file}")

    if not csv_matches_ids(burst_file, data_csv):
        build_burstiness(data_csv, burst_file)
        print(f"Built burstiness: {burst_file}")
    if not csv_matches_ids(struct_file, data_csv):
        build_structure(data_csv, struct_file)
        print(f"Built structure: {struct_file}")

    merge_features(data_csv, feature_dir, feature_dir / "all_features.csv")
    return feature_dir / "all_features.csv"


def build_probability_set(
    data_csv: Path,
    feature_dir: Path,
    model_specs: list[dict[str, str | bool]],
    *,
    max_length: int,
    device: str,
    skip_missing_models: bool,
) -> tuple[list[str], list[str]]:
    used = []
    skipped = []
    for spec in model_specs:
        model_name = str(spec["model_name"])
        model_key = str(spec["model_key"])
        output_file = feature_dir / f"probability_{model_name}.csv"
        if csv_matches_ids(output_file, data_csv):
            print(f"Reused probability: {output_file}")
            used.append(model_name)
            continue
        if not bool(spec["ready"]):
            message = f"Warning: local model missing for {model_name} at {spec['local_dir']}"
            if skip_missing_models:
                print(message)
                skipped.append(model_name)
                continue
            raise FileNotFoundError(message)
        build_probability_features(
            data_csv,
            output_file,
            model_key=model_key,
            local_files_only=True,
            auto_download=False,
            max_length=max_length,
            device_map=model_device_map(device),
        )
        print(f"Built probability: {output_file}")
        used.append(model_name)
    return used, skipped


def maybe_reuse_combined_probability(
    name: str,
    data_csv: Path,
    feature_dir: Path,
    combined_feature_dir: Path,
    model_specs: list[dict[str, str | bool]],
) -> list[str]:
    if name == "combined_public" or not combined_feature_dir.exists():
        return []
    reused = []
    for spec in model_specs:
        model_name = str(spec["model_name"])
        source_file = combined_feature_dir / f"probability_{model_name}.csv"
        target_file = feature_dir / f"probability_{model_name}.csv"
        if csv_matches_ids(target_file, data_csv):
            reused.append(model_name)
            continue
        if subset_feature_csv(source_file, target_file, data_csv) and csv_matches_ids(target_file, data_csv):
            print(f"Subset-reused probability for {name}: {target_file}")
            reused.append(model_name)
    return reused


def maybe_reuse_combined_scale_response(
    name: str,
    data_csv: Path,
    feature_dir: Path,
    combined_feature_dir: Path,
) -> list[str]:
    if name == "combined_public" or not combined_feature_dir.exists():
        return []
    source_file = combined_feature_dir / "scale_response_features.csv"
    target_file = feature_dir / "scale_response_features.csv"
    manifest_source = combined_feature_dir / "scale_response_manifest.json"
    manifest_target = feature_dir / "scale_response_manifest.json"
    if csv_matches_ids(target_file, data_csv) and manifest_target.exists():
        return scale_response_scales(feature_dir)
    if subset_feature_csv(source_file, target_file, data_csv) and csv_matches_ids(target_file, data_csv):
        if manifest_source.exists():
            shutil.copy2(manifest_source, manifest_target)
        print(f"Subset-reused scale_response for {name}: {target_file}")
        return scale_response_scales(feature_dir)
    return []


def build_scale_response_if_requested(data_csv: Path, feature_dir: Path) -> list[str]:
    output = feature_dir / "scale_response_features.csv"
    if csv_matches_ids(output, data_csv) and (feature_dir / "scale_response_manifest.json").exists():
        print(f"Reused scale_response: {output}")
    else:
        build_scale_response_features(feature_dir, output)
        print(f"Built scale_response: {output}")
    return scale_response_scales(feature_dir)


def run_external_eval(checkpoint_dir: Path, test_feature_file: Path, test_csv: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate_external_test.py",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--test_feature_file",
        str(test_feature_file),
        "--test_csv",
        str(test_csv),
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def log_error(results_dir: Path, message: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "error_log.txt"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def summarize_external_results(
    *,
    external_results_root: Path,
    checkpoints_root: Path,
    train_dir: Path,
    train_sets: list[str],
    external_suffix: str,
    summary_name: str,
    external_rows: int,
    feature_mode: str,
    output_suffix: str,
) -> Path | None:
    rows = []
    for name in train_sets:
        metrics_path = external_results_root / f"{name}_on_{external_suffix}" / "detector_metrics.csv"
        metadata_path = checkpoints_root / f"{name}_{output_suffix}" / "train_metadata.json"
        if not metadata_path.exists() and output_suffix == "basic":
            metadata_path = checkpoints_root / name / "train_metadata.json"
        train_csv = train_dir / f"{name}_train.csv"
        if not metrics_path.exists() or not metadata_path.exists() or not train_csv.exists():
            continue
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        train_df = pd.read_csv(train_csv)
        rows.append(
            {
                "train_set": name,
                "feature_mode": feature_mode,
                "n_train_total": int(len(train_df)),
                "n_train_human": int((pd.to_numeric(train_df["label"], errors="coerce") == 0).sum()),
                "n_train_ai": int((pd.to_numeric(train_df["label"], errors="coerce") == 1).sum()),
                "n_features": int(metadata.get("n_features", 0)),
                "best_model": metadata.get("best_model_name"),
                "probability_models_used": "|".join(metadata.get("probability_models_used", [])),
                "scale_response_scales_used": "|".join(metadata.get("available_scale_response_scales", [])),
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "f1": metrics.get("f1"),
                "tpr_at_fpr_1pct": metrics.get("tpr_at_fpr_1pct"),
                "tpr_at_fpr_5pct": metrics.get("tpr_at_fpr_5pct"),
                "fpr_at_tpr_95pct": metrics.get("fpr_at_tpr_95pct"),
                "ece": metrics.get("ECE", metrics.get("expected_calibration_error")),
                "brier_score": metrics.get("brier_score"),
                "external_test_rows": external_rows,
                "results_dir": str(external_results_root / f"{name}_on_{external_suffix}"),
                "checkpoint_dir": str(metadata_path.parent),
            }
        )
    if not rows:
        return None
    summary_path = external_results_root / summary_name
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def maybe_write_comparison(external_results_root: Path) -> Path | None:
    basic_path = external_results_root / "external_eval_summary_basic.csv"
    full_path = external_results_root / "external_eval_summary_full_allfeatures.csv"
    if not basic_path.exists() or not full_path.exists():
        return None
    basic = pd.read_csv(basic_path).add_prefix("basic_")
    full = pd.read_csv(full_path).add_prefix("full_allfeatures_")
    merged = basic.merge(
        full,
        left_on="basic_train_set",
        right_on="full_allfeatures_train_set",
        how="inner",
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "train_set": row["basic_train_set"],
                "basic_auroc": row["basic_auroc"],
                "full_allfeatures_auroc": row["full_allfeatures_auroc"],
                "delta_auroc": row["full_allfeatures_auroc"] - row["basic_auroc"],
                "basic_auprc": row["basic_auprc"],
                "full_allfeatures_auprc": row["full_allfeatures_auprc"],
                "delta_auprc": row["full_allfeatures_auprc"] - row["basic_auprc"],
                "basic_f1": row["basic_f1"],
                "full_allfeatures_f1": row["full_allfeatures_f1"],
                "delta_f1": row["full_allfeatures_f1"] - row["basic_f1"],
                "basic_tpr_at_fpr_1pct": row["basic_tpr_at_fpr_1pct"],
                "full_allfeatures_tpr_at_fpr_1pct": row["full_allfeatures_tpr_at_fpr_1pct"],
                "delta_tpr_at_fpr_1pct": row["full_allfeatures_tpr_at_fpr_1pct"] - row["basic_tpr_at_fpr_1pct"],
                "basic_ece": row["basic_ece"],
                "full_allfeatures_ece": row["full_allfeatures_ece"],
                "delta_ece": row["full_allfeatures_ece"] - row["basic_ece"],
            }
        )
    out = external_results_root / "external_eval_summary_basic_vs_full_allfeatures.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train_sets"
    external_test = Path(args.external_test)
    if not external_test.is_absolute():
        external_test = root / external_test
    external_prepared = ensure_external_prepared(root, external_test)
    feature_root = root / "features_by_dataset"
    results_root = root / "results_by_dataset"
    checkpoints_root = root / "checkpoints"
    external_results_root = root / "results_external"
    full_run = args.max_train_rows is None and args.max_test_rows is None
    output_suffix = args.output_suffix or ("full_basic" if args.feature_mode == "basic" and full_run else args.feature_mode)
    external_base = external_prepared.stem.replace("_prepared", "")
    external_suffix = f"{external_base}_{output_suffix}" if output_suffix else external_base
    external_feature_root = root / "features_external" / external_suffix
    summary_name = f"external_eval_summary_{output_suffix}.csv"

    ordered_train_sets = list(args.train_sets)
    if args.feature_mode == "allfeatures" and "combined_public" in ordered_train_sets:
        ordered_train_sets = ["combined_public"] + [name for name in ordered_train_sets if name != "combined_public"]

    print("Readiness check")
    missing = False
    for name in ordered_train_sets:
        csv_path = train_dir / f"{name}_train.csv"
        if not csv_path.exists():
            missing = True
            print(f"Missing train set: {csv_path}")
        else:
            stats = dataset_stats(csv_path)
            print(f"Train set {name}: rows={stats['rows']} label0={stats['label_0']} label1={stats['label_1']}")
            has_both_classes(csv_path)

    if not external_prepared.exists():
        missing = True
        print(f"Missing external prepared CSV: {external_prepared}")
    else:
        stats = dataset_stats(external_prepared)
        print(f"External test: rows={stats['rows']} label0={stats['label_0']} label1={stats['label_1']}")

    model_specs = resolve_model_specs(args.models)
    print("Probability models:")
    for spec in model_specs:
        print(f"  {spec['model_name']}: {'READY' if spec['ready'] else 'MISSING'} -> {spec['local_dir']}")
    if args.batch_size != 1:
        print(f"Warning: batch_size={args.batch_size} requested, but probability builder is currently row-wise; proceeding without batching.")

    if not any([args.run_features, args.run_train, args.run_external_eval, args.run_probability, args.run_scale_response]):
        if missing:
            print("Next commands:")
            print("python scripts/build_train_splits.py --input data/dataset_english_v1.csv --output_dir data/train_sets")
            print("python scripts/prepare_external_test.py --input data/test/all_samples.json --output data/test/all_samples_prepared.csv")
        for name in ordered_train_sets:
            feature_dir_name = f"{name}_{output_suffix}" if output_suffix else name
            feature_file = feature_root / feature_dir_name / "all_features.csv"
            print(f"{name} feature file exists: {feature_file.exists()} -> {feature_file}")
        external_feature_file = external_feature_root / "all_features.csv"
        print(f"external feature file exists: {external_feature_file.exists()} -> {external_feature_file}")
        print("Expected output directories:")
        for name in ordered_train_sets:
            print(f"  features: {feature_root / (name + ('_' + output_suffix if output_suffix else ''))}")
            print(f"  results: {results_root / (name + ('_' + output_suffix if output_suffix else ''))}")
            print(f"  ckpt: {checkpoints_root / (name + ('_' + output_suffix if output_suffix else ''))}")
        print(f"  external features: {external_feature_root}")
        return

    external_test_csv = maybe_sample_csv(external_prepared, args.max_test_rows, args.seed)
    feature_plan = [
        "burstiness_features.csv",
        "structure_features.csv",
        "probability_qwen25_1_5b.csv",
        "probability_qwen25_7b.csv",
        "probability_qwen25_14b.csv",
        "probability_qwen25_32b.csv",
        "scale_response_features.csv",
        "scale_response_manifest.json",
        "all_features.csv",
    ]
    print("Feature files to generate or reuse:")
    for name in args.train_sets:
        print(f"  {name}: {feature_root / (name + ('_' + output_suffix if output_suffix else ''))}")
    print(f"  external: {external_feature_root}")
    print("Planned files:")
    for item in feature_plan:
        print(f"  - {item}")

    if args.run_features:
        basic_source = root / "features_external" / ("all_samples_full_basic" if full_run else "all_samples")
        build_basic_features(
            external_test_csv,
            external_feature_root,
            reuse_existing_features=args.reuse_existing_features,
            basic_source_dir=basic_source if basic_source.exists() else None,
        )
        if args.feature_mode == "allfeatures" and args.run_probability:
            used, skipped = build_probability_set(
                external_test_csv,
                external_feature_root,
                model_specs,
                max_length=args.max_length,
                device=args.device,
                skip_missing_models=args.skip_missing_models,
            )
            if skipped:
                print(f"External skipped models: {skipped}")
        if args.feature_mode == "allfeatures" and args.run_scale_response:
            scales = build_scale_response_if_requested(external_test_csv, external_feature_root)
            print(f"External scale_response scales: {scales}")
        merge_features(external_test_csv, external_feature_root, external_feature_root / "all_features.csv")

    combined_feature_dir = feature_root / f"combined_public_{output_suffix}"
    for name in ordered_train_sets:
        train_csv = train_dir / f"{name}_train.csv"
        if not train_csv.exists() or not has_both_classes(train_csv):
            continue
        sampled_train_csv = maybe_sample_csv(train_csv, args.max_train_rows, args.seed)
        experiment_name = f"{name}_{output_suffix}" if output_suffix else name
        feature_dir = feature_root / experiment_name
        results_dir = results_root / experiment_name
        checkpoint_dir = checkpoints_root / experiment_name
        feature_file = feature_dir / "all_features.csv"
        probability_used: list[str] = []
        scale_scales: list[str] = []

        try:
            if args.run_features:
                basic_source = feature_root / name
                build_basic_features(
                    sampled_train_csv,
                    feature_dir,
                    reuse_existing_features=args.reuse_existing_features,
                    basic_source_dir=basic_source if basic_source.exists() else None,
                )
                if args.feature_mode == "allfeatures" and args.run_probability:
                    probability_used = maybe_reuse_combined_probability(
                        name,
                        sampled_train_csv,
                        feature_dir,
                        combined_feature_dir,
                        model_specs,
                    )
                    remaining_specs = [spec for spec in model_specs if str(spec["model_name"]) not in probability_used]
                    used_now, skipped = build_probability_set(
                        sampled_train_csv,
                        feature_dir,
                        remaining_specs,
                        max_length=args.max_length,
                        device=args.device,
                        skip_missing_models=args.skip_missing_models,
                    )
                    probability_used.extend(used_now)
                    if skipped:
                        print(f"{name} skipped models: {skipped}")
                if args.feature_mode == "allfeatures" and args.run_scale_response:
                    scale_scales = maybe_reuse_combined_scale_response(
                        name,
                        sampled_train_csv,
                        feature_dir,
                        combined_feature_dir,
                    )
                    if not scale_scales:
                        scale_scales = build_scale_response_if_requested(sampled_train_csv, feature_dir)
                merge_features(sampled_train_csv, feature_dir, feature_file)
            elif not feature_file.exists():
                print(f"Missing feature file for {name}: {feature_file}")
                print("Next step: rerun with --run_features")
                continue
            else:
                probability_used = [spec["model_name"] for spec in model_specs if (feature_dir / f"probability_{spec['model_name']}.csv").exists()]
                scale_scales = scale_response_scales(feature_dir)

            if args.run_train:
                if args.resume and checkpoint_dir.joinpath("best_model.joblib").exists():
                    print(f"Resume enabled; reusing checkpoint for {name}: {checkpoint_dir}")
                else:
                    metadata = {
                        "feature_mode": output_suffix,
                        "feature_groups_used": feature_groups_used(feature_file),
                        "probability_models_used": probability_used,
                        "available_scale_response_scales": scale_scales,
                    }
                    train_and_evaluate(
                        feature_file,
                        results_dir,
                        data_csv=sampled_train_csv,
                        feature_file=feature_file,
                        checkpoint_dir=checkpoint_dir,
                        experiment_name=experiment_name,
                        save_model=True,
                        extra_metadata=metadata,
                    )
                    print(f"Saved checkpoint for {name}: {checkpoint_dir}")

            if args.run_external_eval:
                if not checkpoint_dir.joinpath("best_model.joblib").exists():
                    print(f"Missing checkpoint for {name}: {checkpoint_dir / 'best_model.joblib'}")
                    continue
                external_feature_file = external_feature_root / "all_features.csv"
                if not external_feature_file.exists():
                    print(f"Missing external feature file: {external_feature_file}")
                    print("Next step: rerun with --run_features")
                    continue
                run_external_eval(
                    checkpoint_dir,
                    external_feature_file,
                    external_test_csv,
                    external_results_root / f"{name}_on_{external_suffix}",
                )
                print(f"Finished external eval: {name}_on_{external_suffix}")
        except Exception as exc:
            print(f"Error while processing {name}: {exc}")
            log_error(results_dir, f"[{name}] {exc}\n{traceback.format_exc()}")
            continue

    if args.run_external_eval and full_run:
        summary_path = summarize_external_results(
            external_results_root=external_results_root,
            checkpoints_root=checkpoints_root,
            train_dir=train_dir,
            train_sets=ordered_train_sets,
            external_suffix=external_suffix,
            summary_name=summary_name,
            external_rows=int(len(pd.read_csv(external_prepared))),
            feature_mode=output_suffix,
            output_suffix=output_suffix,
        )
        if summary_path is not None:
            print(f"Saved summary: {summary_path}")
        if output_suffix == "full_allfeatures":
            comparison_path = maybe_write_comparison(external_results_root)
            if comparison_path is not None:
                print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
