"""Prediction CLI for new text."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from . import config
    from .feature_burstiness import extract_burstiness_features
    from .feature_probability import build_probability_features, token_loss_features, load_causal_lm
    from .feature_structure import extract_structure_features
    from .utils import load_json, model_safe_name, top_numeric_deviations
except ImportError:
    import config
    from feature_burstiness import extract_burstiness_features
    from feature_probability import token_loss_features, load_causal_lm
    from feature_structure import extract_structure_features
    from utils import load_json, model_safe_name, top_numeric_deviations


def lightweight_features(text: str) -> dict[str, float]:
    feats = {f"burst_{k}": v for k, v in extract_burstiness_features(text).items()}
    feats.update({f"struct_{k}": v for k, v in extract_structure_features(text).items()})
    return feats


def probability_features(text: str, model_name: str, dtype: str, max_length: int, device_map: str | None, load_4bit: bool) -> dict[str, float]:
    prefix = model_safe_name(model_name)
    tok, model = load_causal_lm(model_name, dtype=dtype, device_map=device_map, load_4bit=load_4bit)
    return token_loss_features(text, tok, model, max_length=max_length, prefix=prefix)


def risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.70:
        return "Medium"
    return "High"


def evidence_messages(row: pd.Series, importance: pd.DataFrame | None, train_ref: pd.DataFrame | None) -> list[str]:
    messages = []
    if row.get("struct_template_phrase_ratio", 0) > 0:
        messages.append("模板化表达比例较高。")
    if row.get("burst_std_sentence_length", 0) < 3 and row.get("burst_sentence_count", 0) >= 2:
        messages.append("句长标准差较低，文本节奏较平滑。")
    loss_std_cols = [c for c in row.index if c.endswith("_loss_std")]
    if any(float(row.get(c, 0) or 0) < 1.5 for c in loss_std_cols):
        messages.append("token-level loss 波动较小。")
    if row.get("burst_compression_ratio_zlib", 1) < 0.8:
        messages.append("压缩率较高，说明文本可能具有较强规则性。")
    if any(c.endswith("_ppl_ratio") for c in row.index):
        messages.append("双模型 PPL ratio 可作为 AI-like pattern 的对比证据。")
    if train_ref is not None:
        for _, col, val in top_numeric_deviations(row, train_ref, importance, 3):
            messages.append(f"{col}={val:.4g} 是较突出的证据特征。")
    return messages[:5] or ["未发现单一强证据，结果主要来自多特征分类器。"]


FEATURE_FAMILY_PREFIXES = {
    "burst": ("burst_",),
    "struct": ("struct_",),
    "probability": ("qwen25_", "bino_"),
    "scale_response": ("scale_",),
    "transition": ("transition_", "trans_"),
    "strict_koopman": ("strict_koopman_", "text_koopman_"),
}


def feature_family(name: str) -> str:
    for family, prefixes in FEATURE_FAMILY_PREFIXES.items():
        if name.startswith(prefixes):
            return family
    return "other"


def feature_compatibility_report(columns: list[str], feats: dict[str, float]) -> dict[str, object]:
    missing = [c for c in columns if c not in feats]
    generated = [c for c in columns if c in feats]
    missing_families = sorted({feature_family(c) for c in missing if feature_family(c) != "other"})
    required_families = sorted({feature_family(c) for c in columns if feature_family(c) != "other"})
    generated_families = sorted({feature_family(c) for c in generated if feature_family(c) != "other"})
    missing_by_family = {}
    for col in missing:
        family = feature_family(col)
        missing_by_family[family] = missing_by_family.get(family, 0) + 1
    return {
        "required_features": len(columns),
        "generated_required_features": len(generated),
        "missing_features": len(missing),
        "missing_feature_fraction": (len(missing) / len(columns)) if columns else 0.0,
        "required_feature_families": required_families,
        "generated_feature_families": generated_families,
        "missing_feature_families": missing_families,
        "missing_by_family": missing_by_family,
        "missing_feature_examples": missing[:25],
    }


def predict_text(
    text: str,
    model_path: str | Path,
    columns_path: str | Path,
    use_lm_features: bool = False,
    lm_model: str = config.DEFAULT_SMALL_MODEL,
    dtype: str = "bfloat16",
    max_length: int = 1024,
    device_map: str | None = None,
    load_4bit: bool = False,
    strict_feature_check: bool = False,
) -> dict[str, object]:
    model = joblib.load(model_path)
    columns = load_json(columns_path, [])
    medians = load_json(Path(columns_path).with_name("feature_medians.json"), {})
    feats = lightweight_features(text)
    if use_lm_features:
        try:
            feats.update(probability_features(text, lm_model, dtype, max_length, device_map, load_4bit))
        except Exception as exc:
            print(f"Warning: LM features skipped: {exc}")
    compatibility = feature_compatibility_report(columns, feats)
    too_many_missing = float(compatibility["missing_feature_fraction"]) > 0.20
    strict_missing = strict_feature_check and int(compatibility["missing_features"]) > 0
    if too_many_missing or strict_missing:
        return {
            "prediction": None,
            "ai_probability": None,
            "risk_level": "Unavailable",
            "top_evidence": [],
            "feature_compatibility": compatibility,
            "error": "This checkpoint requires features not generated by the current prediction pipeline.",
        }
    row = pd.DataFrame([{c: feats.get(c, medians.get(c, 0.0)) for c in columns}])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0, 1])
    else:
        prob = float(model.predict(row)[0])
    pred = "Yes" if prob >= 0.5 else "No"
    importance_path = Path(model_path).with_name("feature_importance.csv")
    train_ref_path = config.FEATURE_DIR / "all_features.csv"
    importance = pd.read_csv(importance_path) if importance_path.exists() else None
    train_ref = pd.read_csv(train_ref_path) if train_ref_path.exists() else None
    full_row = row.iloc[0]
    return {
        "prediction": pred,
        "ai_probability": prob,
        "risk_level": risk_level(prob),
        "top_evidence": evidence_messages(full_row, importance, train_ref),
        "feature_compatibility": compatibility,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model_path", default=str(config.RESULT_DIR / "best_model.pkl"))
    parser.add_argument("--columns_path", default=str(config.RESULT_DIR / "feature_columns.json"))
    parser.add_argument("--use_lm_features", action="store_true")
    parser.add_argument("--lm_model", default=config.DEFAULT_SMALL_MODEL)
    parser.add_argument("--dtype", default=config.DTYPE)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--strict_feature_check", action="store_true")
    args = parser.parse_args()
    out = predict_text(
        args.text,
        args.model_path,
        args.columns_path,
        args.use_lm_features,
        args.lm_model,
        args.dtype,
        args.max_length,
        args.device_map,
        args.load_4bit,
        args.strict_feature_check,
    )
    if out.get("error"):
        print(out["error"])
        report = out.get("feature_compatibility", {})
        print(f"required_features: {report.get('required_features')}")
        print(f"generated_required_features: {report.get('generated_required_features')}")
        print(f"missing_features: {report.get('missing_features')}")
        print(f"missing_feature_fraction: {report.get('missing_feature_fraction'):.2%}")
        print(f"missing_feature_families: {', '.join(report.get('missing_feature_families', []))}")
        if report.get("missing_feature_examples"):
            print("missing_feature_examples:")
            for col in report["missing_feature_examples"]:
                print(f"- {col}")
        return
    print(f"prediction: {out['prediction']}")
    print(f"ai_probability: {out['ai_probability']:.4f}")
    print(f"risk_level: {out['risk_level']}")
    report = out.get("feature_compatibility", {})
    print(f"missing_features: {report.get('missing_features')} / {report.get('required_features')}")
    if report.get("missing_feature_families"):
        print(f"missing_feature_families: {', '.join(report.get('missing_feature_families', []))}")
    print("top_evidence:")
    for msg in out["top_evidence"]:
        print(f"- {msg}")


if __name__ == "__main__":
    main()
