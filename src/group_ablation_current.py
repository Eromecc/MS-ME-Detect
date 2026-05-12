"""Current feature-group ablation utility for available MS-ME-Detect features."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import config
except ImportError:
    import config


METADATA_COLUMNS = {"id", "text", "label", "type", "source", "topic", "group_id"}
SMALL_DATASET_THRESHOLD = 50

BURSTINESS_FALLBACK_NAMES = {
    "text_length",
    "char_count",
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "std_sentence_length",
    "sentence_length_cv",
    "punctuation_ratio",
    "type_token_ratio",
    "compression_ratio_zlib",
    "compression_ratio_gzip",
    "zipf_deviation_score",
}

STRUCTURE_FALLBACK_NAMES = {
    "template_phrase_count",
    "template_phrase_ratio",
    "transition_phrase_count",
    "transition_phrase_ratio",
    "repeated_bigram_ratio",
    "repeated_trigram_ratio",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "number_density",
    "domain_term_density",
    "content_word_ratio",
}

PERTURBATION_NAME_PARTS = (
    "perturbation_count",
    "avg_length_delta",
    "std_length_delta",
    "avg_compression_ratio_delta",
    "avg_jaccard_similarity",
    "std_jaccard_similarity",
)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def read_feature_group_file(feature_groups_path: Path, group_name: str, available: set[str]) -> list[str]:
    if not feature_groups_path.exists():
        return []
    groups = pd.read_csv(feature_groups_path)
    if not {"feature", "group"}.issubset(groups.columns):
        return []
    selected = groups.loc[groups["group"].astype(str) == group_name, "feature"].astype(str)
    return [feature for feature in selected if feature in available]


def suffix_matches(column: str, names: set[str]) -> bool:
    return column in names or any(column.endswith(f"_{name}") for name in names)


def detect_feature_groups(df: pd.DataFrame, feature_groups_path: Path) -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(df)
    available = set(numeric_cols)

    burstiness = read_feature_group_file(feature_groups_path, "burstiness", available)
    if not burstiness:
        burstiness = [c for c in numeric_cols if c.startswith("burst_") or suffix_matches(c, BURSTINESS_FALLBACK_NAMES)]

    structure = read_feature_group_file(feature_groups_path, "structure", available)
    if not structure:
        structure = [c for c in numeric_cols if c.startswith("struct_") or suffix_matches(c, STRUCTURE_FALLBACK_NAMES)]

    perturbation = [
        c
        for c in numeric_cols
        if c.startswith("pert_") or c.startswith("perturbation_") or any(part in c for part in PERTURBATION_NAME_PARTS)
    ]
    qwen25_1_5b = [c for c in numeric_cols if c.startswith("qwen25_1_5b_")]

    return {
        "burstiness": sorted(set(burstiness), key=numeric_cols.index),
        "structure": sorted(set(structure), key=numeric_cols.index),
        "perturbation": sorted(set(perturbation), key=numeric_cols.index),
        "qwen25_1_5b": qwen25_1_5b,
        "all_current": numeric_cols,
    }


def unique_columns(columns: list[str]) -> list[str]:
    seen = set()
    out = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            out.append(column)
    return out


def build_ablation_groups(feature_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    burstiness = feature_groups["burstiness"]
    structure = feature_groups["structure"]
    perturbation = feature_groups["perturbation"]
    qwen = feature_groups["qwen25_1_5b"]
    return {
        "burstiness_only": burstiness,
        "structure_only": structure,
        "perturbation_only": perturbation,
        "qwen25_1_5b_only": qwen,
        "burstiness_structure": unique_columns(burstiness + structure),
        "burstiness_qwen25_1_5b": unique_columns(burstiness + qwen),
        "structure_qwen25_1_5b": unique_columns(structure + qwen),
        "burstiness_structure_perturbation": unique_columns(burstiness + structure + perturbation),
        "burstiness_structure_qwen25_1_5b": unique_columns(burstiness + structure + qwen),
        "all_current_features": feature_groups["all_current"],
    }


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=config.RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
    }


def split_dataset(df: pd.DataFrame):
    y = df["label"].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Label counts are too small for stratified splitting; using an unstratified split.", RuntimeWarning)
    return train_test_split(df, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify)


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]):
    x_train = train_df[columns].replace([np.inf, -np.inf], np.nan)
    x_test = test_df[columns].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median(numeric_only=True).fillna(0.0)
    return x_train.fillna(medians), x_test.fillna(medians)


def model_probabilities(model, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return np.asarray(scores, dtype=float)
    return None


def evaluate(model_name: str, model, group_name: str, columns: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame, y_train, y_test) -> dict[str, object]:
    x_train, x_test = prepare_xy(train_df, test_df, columns)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model_probabilities(model, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    result: dict[str, object] = {
        "ablation_group": group_name,
        "model": model_name,
        "n_features": len(columns),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if prob is not None and len(set(y_test)) > 1:
        try:
            result["roc_auc"] = roc_auc_score(y_test, prob)
        except ValueError:
            result["roc_auc"] = np.nan
    return result


def write_report(
    report_path: Path,
    results: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    skipped: list[str],
    n_rows: int,
    train_size: int,
    test_size: int,
) -> None:
    top = results.sort_values(["f1", "roc_auc"], ascending=False, na_position="last").head(10)
    lines = [
        "Current Feature-Group Ablation Report",
        "",
        f"Rows: {n_rows}",
        f"Train rows: {train_size}",
        f"Test rows: {test_size}",
        f"Random state: {config.RANDOM_STATE}",
        "",
        "Detected current feature groups:",
    ]
    for name in ["burstiness", "structure", "perturbation", "qwen25_1_5b", "all_current"]:
        lines.append(f"- {name}: {len(feature_groups[name])} features")
    if n_rows < SMALL_DATASET_THRESHOLD:
        lines.extend(
            [
                "",
                f"WARNING: This dataset has only {n_rows} rows. Results are a smoke test only and are too small for real conclusions.",
            ]
        )
    if skipped:
        lines.extend(["", "Skipped ablation groups with zero detected features:"])
        lines.extend(f"- {name}" for name in skipped)
    lines.extend(["", "Top results sorted by f1 and roc_auc:", top.to_string(index=False)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_current_ablation(input_path: str | Path, result_dir: str | Path, feature_groups_path: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(input_path)
    result_dir = Path(result_dir)
    feature_groups_path = Path(feature_groups_path) if feature_groups_path else input_path.parent / "feature_groups.csv"
    result_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError("Input feature file must contain a label column.")
    if len(df) < SMALL_DATASET_THRESHOLD:
        print(f"WARNING: dataset has only {len(df)} rows; this ablation is a smoke test only.")

    feature_groups = detect_feature_groups(df, feature_groups_path)
    ablations = build_ablation_groups(feature_groups)
    train_df, test_df, y_train, y_test = split_dataset(df)

    rows = []
    skipped = []
    for group_name, columns in ablations.items():
        if not columns:
            skipped.append(group_name)
            continue
        for model_name, model in candidate_models().items():
            rows.append(evaluate(model_name, model, group_name, columns, train_df, test_df, y_train, y_test))

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No ablation groups had detected features.")
    results = results.sort_values(["f1", "roc_auc"], ascending=False, na_position="last").reset_index(drop=True)

    results_path = result_dir / "current_group_ablation_results.csv"
    report_path = result_dir / "current_group_ablation_report.txt"
    results.to_csv(results_path, index=False)
    write_report(report_path, results, feature_groups, skipped, len(df), len(train_df), len(test_df))

    print(results.to_string(index=False))
    print(f"\nSaved results to {results_path}")
    print(f"Saved report to {report_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run current feature-group ablations.")
    parser.add_argument("--input", default=str(config.FEATURE_DIR / "all_features.csv"))
    parser.add_argument("--result_dir", default=str(config.RESULT_DIR))
    parser.add_argument("--feature_groups", default=str(config.FEATURE_DIR / "feature_groups.csv"))
    args = parser.parse_args()
    config.ensure_dirs()
    run_current_ablation(args.input, args.result_dir, args.feature_groups)


if __name__ == "__main__":
    main()
