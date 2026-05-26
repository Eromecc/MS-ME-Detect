#!/usr/bin/env python3
"""Validation-only incremental blend of an existing prediction file and probe candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, max_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if np.any(ok) else 0.0


def metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_true, score))
    auprc = float(average_precision_score(y_true, score))
    low = tpr_at_fpr(y_true, score)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_fpr5": low,
        "f1": f1,
        "composite": float(0.45 * auroc + 0.35 * auprc + 0.20 * low),
    }


def score_col(df: pd.DataFrame) -> str:
    for col in ["score", "ai_probability", "probability"]:
        if col in df.columns:
            return col
    raise ValueError(f"no score column in {df.columns.tolist()}")


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = score_col(df)
    keep = [c for c in ["id", "text", "label", "source_dataset", "domain", "generator", "source", "split"] if c in df.columns]
    out = df[keep].copy()
    out["score"] = df[col].astype(float).to_numpy()
    return out


def rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    return (rankdata(x, method="average") - 1.0) / (len(x) - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", required=True)
    parser.add_argument("--base_val", required=True)
    parser.add_argument("--base_all", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha_max", type=float, default=0.2)
    parser.add_argument("--alpha_step", type=float, default=0.005)
    parser.add_argument(
        "--plateau_eps",
        nargs="*",
        type=float,
        default=[],
        help="Optional validation-composite tolerances; select largest alpha within max composite - eps.",
    )
    parser.add_argument("--transforms", nargs="*", default=["raw", "rank"], choices=["raw", "rank"])
    args = parser.parse_args()

    base_val = load_scores(Path(args.base_val))
    base_all = load_scores(Path(args.base_all))
    y_val = base_val["label"].to_numpy(dtype=int)
    y_all = base_all["label"].to_numpy(dtype=int)
    base_val_score = base_val["score"].to_numpy(dtype=float)
    base_all_score = base_all["score"].to_numpy(dtype=float)
    alphas = np.arange(0.0, args.alpha_max + 1e-12, args.alpha_step)
    policies = {"val_auroc": "auroc", "val_composite": "composite", "val_low_fpr": "tpr_at_fpr5"}
    best = {policy: None for policy in policies}
    rows = []

    cand_dir = Path(args.candidate_dir)
    for val_path in sorted(cand_dir.glob("*__validation.csv")):
        all_path = cand_dir / val_path.name.replace("__validation.csv", "__all_samples.csv")
        if not all_path.exists():
            continue
        cand_name = val_path.name.removesuffix("__validation.csv")
        cand_val = load_scores(val_path)
        cand_all = load_scores(all_path)
        merged_val = base_val[["id", "label"]].merge(cand_val[["id", "score"]], on="id", validate="one_to_one", suffixes=("", "_cand"))
        merged_all = base_all[["id", "label"]].merge(cand_all[["id", "score"]], on="id", validate="one_to_one", suffixes=("", "_cand"))
        if len(merged_val) != len(base_val) or len(merged_all) != len(base_all):
            raise ValueError(f"candidate row mismatch: {cand_name}")
        for transform in args.transforms:
            cv = cand_val["score"].to_numpy(dtype=float)
            ca = cand_all["score"].to_numpy(dtype=float)
            bv = base_val_score
            ba = base_all_score
            if transform == "rank":
                cv = rank01(cv)
                ca = rank01(ca)
                bv = rank01(bv)
                ba = rank01(ba)
            for alpha in alphas:
                sv = np.clip((1.0 - alpha) * bv + alpha * cv, 0.0, 1.0)
                sa = np.clip((1.0 - alpha) * ba + alpha * ca, 0.0, 1.0)
                vm = metrics(y_val, sv)
                am = metrics(y_all, sa)
                row = {
                    "base_name": args.base_name,
                    "candidate": f"{cand_name}__{transform}",
                    "alpha": float(alpha),
                    "val_auroc": vm["auroc"],
                    "val_auprc": vm["auprc"],
                    "val_tpr_at_fpr5": vm["tpr_at_fpr5"],
                    "val_f1": vm["f1"],
                    "val_composite": vm["composite"],
                    "all_samples_auroc": am["auroc"],
                    "all_samples_auprc": am["auprc"],
                    "all_samples_tpr_at_fpr5": am["tpr_at_fpr5"],
                    "all_samples_f1": am["f1"],
                    "all_samples_composite": am["composite"],
                }
                rows.append(row)
                for policy, key in policies.items():
                    current = best[policy]
                    if current is None or row[f"val_{key}" if key != "tpr_at_fpr5" else "val_tpr_at_fpr5"] > current[0]:
                        best[policy] = (row[f"val_{key}" if key != "tpr_at_fpr5" else "val_tpr_at_fpr5"], row, sv, sa)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows).sort_values("val_composite", ascending=False)
    grid.to_csv(out_dir / "incremental_probe_grid.csv", index=False)
    for eps in args.plateau_eps:
        max_comp = float(grid["val_composite"].max())
        pool = grid[grid["val_composite"] >= max_comp - float(eps)]
        if pool.empty:
            continue
        row = pool.sort_values(["alpha", "val_composite", "val_auroc"], ascending=[False, False, False]).iloc[0].to_dict()
        cand_name, transform = str(row["candidate"]).rsplit("__", 1)
        val_path = cand_dir / f"{cand_name}__validation.csv"
        all_path = cand_dir / f"{cand_name}__all_samples.csv"
        cand_val = load_scores(val_path)
        cand_all = load_scores(all_path)
        cv = cand_val["score"].to_numpy(dtype=float)
        ca = cand_all["score"].to_numpy(dtype=float)
        bv = base_val_score
        ba = base_all_score
        if transform == "rank":
            cv = rank01(cv)
            ca = rank01(ca)
            bv = rank01(bv)
            ba = rank01(ba)
        alpha = float(row["alpha"])
        sv = np.clip((1.0 - alpha) * bv + alpha * cv, 0.0, 1.0)
        sa = np.clip((1.0 - alpha) * ba + alpha * ca, 0.0, 1.0)
        best[f"val_composite_plateau_eps_{eps:g}_largest_alpha"] = (float(row["val_composite"]), row, sv, sa)
    selected = []
    for policy, result in best.items():
        if result is None:
            continue
        _, row, sv, sa = result
        rec = dict(row)
        rec["selection_policy"] = policy
        rec["leakage_policy"] = "candidate/alpha selected on validation only; all_samples eval only"
        selected.append(rec)
        val_out = base_val.copy()
        all_out = base_all.copy()
        val_out["score"] = sv
        all_out["score"] = sa
        val_out["prediction"] = (sv >= 0.5).astype(int)
        all_out["prediction"] = (sa >= 0.5).astype(int)
        val_out["candidate_id"] = rec["candidate"]
        all_out["candidate_id"] = rec["candidate"]
        val_out.to_csv(out_dir / f"{policy}__validation_predictions.csv", index=False)
        all_out.to_csv(out_dir / f"{policy}__all_samples_predictions.csv", index=False)
    selected_df = pd.DataFrame(selected).sort_values("val_composite", ascending=False)
    selected_df.to_csv(out_dir / "incremental_probe_selected.csv", index=False)
    print(selected_df.to_string(index=False))


if __name__ == "__main__":
    main()
