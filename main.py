"""Main orchestration CLI for MS-ME-Detect."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src import config
from src.feature_binoculars import build_features as build_binoculars
from src.feature_burstiness import build_features as build_burstiness
from src.feature_perturbation import build_features as build_perturbation
from src.feature_probability import build_probability_features
from src.feature_scale_response import build_scale_response_features
from src.feature_structure import build_features as build_structure
from src.group_ablation_current import run_current_ablation
from src.merge_features import merge_features
from src.predict import predict_text
from src.preprocess import load_dataset, train_test_split_dataset
from src.train_eval import train_and_evaluate
from src.utils import model_safe_name


def prepare_data() -> None:
    config.ensure_dirs()
    df = load_dataset(config.DATA_PATH)
    df.to_csv(config.DATA_PATH, index=False)
    train_test_split_dataset(config.DATA_PATH, config.TRAIN_PATH, config.TEST_PATH, config.TEST_SIZE, config.RANDOM_STATE)


def run_lightweight_features(include_perturbation: bool = False) -> None:
    build_burstiness(config.DATA_PATH, config.FEATURE_DIR / "burstiness_features.csv")
    build_structure(config.DATA_PATH, config.FEATURE_DIR / "structure_features.csv")
    if include_perturbation:
        build_perturbation(config.DATA_PATH, config.FEATURE_DIR / "perturbation_features.csv", mode="rule")


def run_probability_features(args) -> None:
    models = [args.small_model, args.medium_model, args.large_model]
    if args.include_32b:
        models.append(args.xl_model)
    for model in models:
        safe = model_safe_name(model)
        out = config.FEATURE_DIR / f"probability_{safe}.csv"
        build_probability_features(
            config.DATA_PATH,
            out,
            model,
            dtype=args.dtype,
            max_length=args.max_length,
            device_map=args.device_map,
            load_4bit=args.load_4bit,
            allow_fallback=args.allow_fallback,
        )


def probability_output_for_key(model_key: str) -> Path:
    return config.FEATURE_DIR / f"probability_{config.MODEL_KEY_PREFIX[model_key]}.csv"


def run_probability_feature_keys(args, models: list[str]) -> None:
    prepare_data()
    for model_key in models:
        build_probability_features(
            config.DATA_PATH,
            probability_output_for_key(model_key),
            model_key=model_key,
            dtype=args.dtype,
            max_length=args.max_length,
            device_map=args.device_map,
            load_4bit=args.load_4bit,
            allow_fallback=args.allow_fallback,
            local_files_only=True,
            auto_download=False,
            model_root=args.model_root,
        )


def run_script(script_name: str, extra_args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parent / "scripts" / script_name
    return subprocess.run([sys.executable, str(script), *extra_args], check=check)


def run_full_qwen(args) -> None:
    prepare_data()
    run_lightweight_features(include_perturbation=args.include_perturbation)
    run_script("check_models.py", ["--models", *args.models])
    run_probability_feature_keys(args, args.models)
    build_scale_response_features(config.FEATURE_DIR, config.FEATURE_DIR / "scale_response_features.csv")
    merge_features(config.DATA_PATH, config.FEATURE_DIR, config.FEATURE_DIR / "all_features.csv")
    train_and_evaluate(config.FEATURE_DIR / "all_features.csv", config.RESULT_DIR)


def run_features(args, include_lm: bool) -> None:
    prepare_data()
    run_lightweight_features(include_perturbation=args.include_perturbation)
    if include_lm:
        run_probability_features(args)
        build_binoculars(
            config.DATA_PATH,
            config.FEATURE_DIR / "binoculars_features.csv",
            args.small_model,
            args.medium_model,
            dtype=args.dtype,
            max_length=args.max_length,
            device_map=args.device_map,
            load_4bit=args.load_4bit,
            include_large_pair=args.include_large_pair,
        )
    build_scale_response_features(config.FEATURE_DIR, config.FEATURE_DIR / "scale_response_features.csv")
    merge_features(config.DATA_PATH, config.FEATURE_DIR, config.FEATURE_DIR / "all_features.csv")


def run_predict(args) -> None:
    out = predict_text(
        args.text,
        config.RESULT_DIR / "best_model.pkl",
        config.RESULT_DIR / "feature_columns.json",
        use_lm_features=args.use_lm_features,
        lm_model=args.predict_model,
        dtype=args.dtype,
        max_length=args.max_length,
        device_map=args.device_map,
        load_4bit=args.load_4bit,
    )
    print(f"prediction: {out['prediction']}")
    print(f"ai_probability: {out['ai_probability']:.4f}")
    print(f"risk_level: {out['risk_level']}")
    print("top_evidence:")
    for msg in out["top_evidence"]:
        print(f"- {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MS-ME-Detect pipeline")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "features",
            "train",
            "predict",
            "demo",
            "scale_response",
            "download_models",
            "check_models",
            "probability",
            "full_qwen",
            "current_ablation",
        ],
        default="demo",
    )
    parser.add_argument("--text", default="")
    parser.add_argument("--models", nargs="+", default=["small", "medium", "large"], choices=list(config.MODEL_REGISTRY))
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--download_backend", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--small_model", default=config.DEFAULT_SMALL_MODEL)
    parser.add_argument("--medium_model", default=config.DEFAULT_MEDIUM_MODEL)
    parser.add_argument("--large_model", default=config.DEFAULT_LARGE_MODEL)
    parser.add_argument("--xl_model", default=config.DEFAULT_XL_MODEL)
    parser.add_argument("--predict_model", default=config.DEFAULT_SMALL_MODEL)
    parser.add_argument("--include_32b", action="store_true")
    parser.add_argument("--include_large_pair", action="store_true")
    parser.add_argument("--include_perturbation", action="store_true")
    parser.add_argument("--dtype", default=config.DTYPE)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--allow_fallback", action="store_true")
    parser.add_argument("--use_lm_features", action="store_true")
    parser.add_argument("--skip_lm", action="store_true", help="Skip Qwen LM features in all/features mode.")
    args = parser.parse_args()

    config.ensure_dirs()
    if args.mode == "demo":
        run_features(args, include_lm=False)
        print(train_and_evaluate(config.FEATURE_DIR / "all_features.csv", config.RESULT_DIR))
    elif args.mode == "features":
        run_features(args, include_lm=not args.skip_lm)
    elif args.mode == "train":
        train_and_evaluate(config.FEATURE_DIR / "all_features.csv", config.RESULT_DIR)
    elif args.mode == "scale_response":
        build_scale_response_features(config.FEATURE_DIR, config.FEATURE_DIR / "scale_response_features.csv")
    elif args.mode == "download_models":
        extra = ["--models", *args.models, "--backend", args.download_backend]
        if args.model_root:
            extra.extend(["--model_root", args.model_root])
        run_script("download_models.py", extra)
    elif args.mode == "check_models":
        extra = ["--models", *args.models]
        if args.model_root:
            extra.extend(["--model_root", args.model_root])
        result = run_script("check_models.py", extra, check=False)
        raise SystemExit(result.returncode)
    elif args.mode == "probability":
        run_probability_feature_keys(args, args.models)
    elif args.mode == "full_qwen":
        run_full_qwen(args)
    elif args.mode == "current_ablation":
        run_current_ablation(config.FEATURE_DIR / "all_features.csv", config.RESULT_DIR)
    elif args.mode == "predict":
        if not args.text:
            raise ValueError("--text is required for prediction mode.")
        run_predict(args)
    elif args.mode == "all":
        run_features(args, include_lm=not args.skip_lm)
        train_and_evaluate(config.FEATURE_DIR / "all_features.csv", config.RESULT_DIR)


if __name__ == "__main__":
    main()
