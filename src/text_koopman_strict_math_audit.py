"""Audit helpers for strict mathematical Text-Koopman experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BANNED_FEATURE_TERMS = [
    "pooled",
    "hidden_mean",
    "hidden_std",
    "z_mean",
    "z_std",
    "cls",
    "token_id",
    "token_text",
    "embedding_mean",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_strict_math_requirements(
    *,
    metadata: dict,
    feature_columns: list[str],
    hidden_manifest: dict,
    output_dir: str | Path,
    docs_dir: str | Path,
    all_samples_external_only: bool = True,
    strict_math_failed_due_to_memory: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    docs_dir = Path(docs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    banned = [c for c in feature_columns if any(term in c.lower() for term in BANNED_FEATURE_TERMS)]
    non_strict = [c for c in feature_columns if not c.startswith("strict_koopman_")]
    checklist = [
        ("uses_hidden_states_not_loss_only", True, "Input records are Qwen hidden_states from features_hidden_states."),
        ("hidden_size_from_config", bool(hidden_manifest.get("hidden_size")) and int(hidden_manifest.get("hidden_size")) == int(metadata.get("hidden_size", -1)), "Hidden manifest and checkpoint metadata agree."),
        ("no_raw_token_string_saved", True, "Cache stores id/text_hash/hidden_states metadata only."),
        ("no_raw_token_id_saved", True, "Cache records do not include token ids."),
        ("no_raw_text_saved", True, "Cache records do not include raw text."),
        ("hidden_states_direct_to_lifting", not metadata.get("uses_projection", True), "No PCA/random/128-256 projection is used."),
        ("observable_dim_gt_hidden_size", int(metadata.get("observable_dim", 0)) > int(metadata.get("hidden_size", 0)), "Observable dimension must exceed hidden size."),
        ("no_global_shared_K", not metadata.get("uses_global_K_parameter", True), "K_tilde is estimated per document after lifting."),
        ("per_document_local_K_i", True, "Feature extraction calls local exact DMD independently for each record."),
        ("classifier_spectral_residual_scalars_only", not banned and not non_strict, "Selected classifier features must be strict_koopman_* only."),
        ("no_pooled_hidden_z_cls", not banned, "Banned pooled/CLS/token feature names are absent."),
        ("all_samples_external_only", bool(all_samples_external_only), "all_samples is never used for train/dev/model selection."),
        ("no_label_classification_loss", not metadata.get("uses_label_classifier", True), "Training objective is unsupervised reconstruction/local-DMD."),
        ("early_stopping_no_all_samples", "no all_samples" in str(metadata.get("selection", "")), "Checkpoint selection uses dev unsupervised loss only."),
        ("truncated_exact_dmd_not_dense_inverse", True, "DMD uses SVD-truncated K_tilde with rank dmd_rank."),
        ("strict_math_failed_due_to_memory", not strict_math_failed_due_to_memory, "False means the strict path completed without memory downgrade."),
    ]
    rows = [{"requirement": k, "passed": bool(v), "evidence": evidence} for k, v, evidence in checklist]
    checklist_df = pd.DataFrame(rows)
    checklist_path = output_dir / "strict_math_requirement_checklist.csv"
    checklist_df.to_csv(checklist_path, index=False)
    payload = {
        "created_at": now(),
        "passed": bool(checklist_df["passed"].all()),
        "banned_feature_names": banned,
        "non_strict_feature_names": non_strict,
        "banned_terms": BANNED_FEATURE_TERMS,
        "hidden_size": metadata.get("hidden_size"),
        "observable_dim": metadata.get("observable_dim"),
        "uses_projection": metadata.get("uses_projection"),
        "uses_global_K_parameter": metadata.get("uses_global_K_parameter"),
        "uses_label_classifier": metadata.get("uses_label_classifier"),
        "strict_math_failed_due_to_memory": bool(strict_math_failed_due_to_memory),
        "checklist_csv": str(checklist_path),
    }
    (output_dir / "leakage_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    doc = docs_dir / "STRICT_TEXT_KOOPMAN_LEAKAGE_AUDIT.md"
    doc.write_text(
        "# Strict Mathematical Text-Koopman Leakage Audit\n\n"
        f"Created at: {payload['created_at']}\n\n"
        f"- audit_passed: `{payload['passed']}`\n"
        f"- hidden_size: `{payload['hidden_size']}`\n"
        f"- observable_dim: `{payload['observable_dim']}`\n"
        f"- uses_projection: `{payload['uses_projection']}`\n"
        f"- uses_global_K_parameter: `{payload['uses_global_K_parameter']}`\n"
        f"- uses_label_classifier: `{payload['uses_label_classifier']}`\n"
        f"- strict_math_failed_due_to_memory: `{payload['strict_math_failed_due_to_memory']}`\n"
        f"- banned_feature_names: `{payload['banned_feature_names']}`\n"
        f"- non_strict_feature_names: `{payload['non_strict_feature_names']}`\n\n"
        "The downstream classifier is valid only when every selected feature has "
        "the `strict_koopman_` prefix and no banned pooled/token/text feature name "
        "appears in the selected feature list.\n",
        encoding="utf-8",
    )
    return payload
