#!/usr/bin/env python3
"""Deep DMD / Koopman encoder experiment runner."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_transition_formal_experiment import cleaned_full_columns, eval_one, fit_select, make_xy, merge_full_features, save_fig  # noqa: E402
from run_transition_fullscale_optimized import build_base_datasets, composite_train_dev  # noqa: E402
from run_koopman_dmd_experiment import merge_transition  # noqa: E402
from src.deep_dmd_dataset import (  # noqa: E402
    DeepDMDTokenDataset,
    fit_loss_bins,
    fit_observable_scaler,
    read_token_loss_cache,
)
from src.deep_dmd_features import extract_deep_dmd_features  # noqa: E402
from src.deep_dmd_model import DeepDMDEncoder  # noqa: E402
from src.deep_dmd_train import evaluate_deep_dmd, predict_scores, train_deep_dmd  # noqa: E402
from src.train_eval import detector_metrics, probabilities, save_calibration_curve, save_pr_curve, save_roc_curve  # noqa: E402
from src.utils import write_csv  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def token_cache_path(model_name: str, dataset: str) -> Path:
    return ROOT / "features_token_loss" / model_name / f"{dataset}_token_loss.jsonl.gz"


def base_names() -> list[str]:
    return ["m4_train", "m4_dev", "m4_test", "ghostbuster_train", "ghostbuster_dev", "ghostbuster_test", "hc3_plus_train", "hc3_plus_dev", "hc3_plus_test", "all_samples"]


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rank_tuple(m: dict) -> tuple[float, float, float, float]:
    return tuple(float(m.get(k, -np.inf)) if pd.notna(m.get(k, np.nan)) else -np.inf for k in ["auprc", "auroc", "tpr_at_fpr_5pct", "f1"])


def cache_for_model(model_name: str) -> dict[str, dict[str, dict]]:
    caches = {}
    for name in base_names():
        path = token_cache_path(model_name, name)
        if path.exists():
            caches[name] = read_token_loss_cache(path)
    return caches


def lookup_cache(caches: dict[str, dict], rid: str) -> dict | None:
    for cache in caches.values():
        if rid in cache:
            return cache[rid]
    return None


def make_union_cache(caches: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for cache in caches.values():
        out.update(cache)
    return out


def build_deep_datasets(data: dict[str, pd.DataFrame], train_name: str, model_name: str, max_seq_len: int, min_tokens: int):
    train_meta, dev_meta = composite_train_dev(data, train_name)
    caches = cache_for_model(model_name)
    union = make_union_cache(caches)
    train_ids = [str(x) for x in train_meta["id"]]
    bins = fit_loss_bins(union, train_ids, n_states=5)
    scaler = fit_observable_scaler(union, train_ids, max_seq_len=max_seq_len, loss_bins=bins)
    datasets = {
        "train": DeepDMDTokenDataset(train_meta, union, max_seq_len=max_seq_len, loss_bins=bins, scaler=scaler, min_tokens=min_tokens),
        "dev": DeepDMDTokenDataset(dev_meta, union, max_seq_len=max_seq_len, loss_bins=bins, scaler=scaler, min_tokens=min_tokens),
    }
    for name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
        datasets[name] = DeepDMDTokenDataset(data[name], union, max_seq_len=max_seq_len, loss_bins=bins, scaler=scaler, min_tokens=min_tokens)
    return datasets, scaler, bins


def save_score_eval(pred: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pred.copy()
    pred["prediction"] = (pred["deep_dmd_score"] >= 0.5).astype(int)
    m = detector_metrics(pred["label"].astype(int), pred["deep_dmd_score"], y_pred=pred["prediction"])
    write_csv(pred, out_dir / "predictions.csv")
    write_csv(pd.DataFrame([m]), out_dir / "detector_metrics.csv")
    write_csv(pd.DataFrame([m]), out_dir / "metrics.csv")
    write_csv(save_roc_curve(pred["label"], pred["deep_dmd_score"], out_dir), out_dir / "roc_curve.csv")
    write_csv(save_pr_curve(pred["label"], pred["deep_dmd_score"], out_dir), out_dir / "pr_curve.csv")
    write_csv(save_calibration_curve(pred["label"], pred["deep_dmd_score"], out_dir), out_dir / "calibration_bins.csv")
    return m


def choose_alpha(y, a, b):
    rows = []
    best = None
    for alpha in np.linspace(0, 1, 11):
        score = alpha * a + (1 - alpha) * b
        m = detector_metrics(y, score)
        rows.append({"alpha": float(alpha), **m})
        if best is None or rank_tuple(m) > best[0]:
            best = (rank_tuple(m), float(alpha))
    return best[1], rows


def threshold_for_fpr(y_true, prob, target):
    best = (1.0, 0.0, 0.0)
    for thr in np.unique(np.r_[0.0, 1.0, prob]):
        pred = (prob >= thr).astype(int)
        m = detector_metrics(y_true, prob, y_pred=pred, threshold=float(thr))
        if m["fpr"] <= target + 1e-12 and m["tpr"] >= best[1]:
            best = (float(thr), float(m["tpr"]), float(m["fpr"]))
    return best


LOSS_CONFIGS = {
    "A": {"lambda_pred": 1.0, "lambda_recon": 0.2, "lambda_cls": 0.5, "lambda_reg": 1e-4, "lambda_stability": 0.1},
    "B": {"lambda_pred": 1.0, "lambda_recon": 0.1, "lambda_cls": 1.0, "lambda_reg": 1e-4, "lambda_stability": 0.1},
    "C": {"lambda_pred": 2.0, "lambda_recon": 0.2, "lambda_cls": 0.5, "lambda_reg": 1e-4, "lambda_stability": 0.2},
}


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "").replace("-", "m")


def load_if_available(model: DeepDMDEncoder, ckpt: Path, device: torch.device) -> bool:
    path = ckpt / "deep_dmd_model.pt"
    if not path.exists():
        return False
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return True


def plot_latent_probe(latent_df: pd.DataFrame, out_dir: Path, prefix: str) -> dict:
    cols = [c for c in latent_df.columns if c.startswith("latent_")]
    if not cols or len(latent_df) < 10:
        return {}
    x = latent_df[cols].fillna(0.0)
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(x))
    p = latent_df.copy()
    p["PC1"] = coords[:, 0]
    p["PC2"] = coords[:, 1]
    p["label_name"] = p["label"].map({0: "Human", 1: "AI"})
    for hue, stem in [("label_name", "latent_pca_by_label"), ("source_dataset", "latent_pca_by_source"), ("domain", "latent_pca_by_domain")]:
        if hue in p.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=p, x="PC1", y="PC2", hue=hue, s=22, alpha=0.65, ax=ax)
            ax.set_title(f"Deep DMD latent PCA by {hue}")
            save_fig(fig, out_dir / f"{prefix}_{stem}")
    probes = {}
    for name, col in [("label", "label"), ("source", "source_dataset"), ("domain", "domain")]:
        if col in latent_df.columns and latent_df[col].nunique() > 1:
            y = latent_df[col].astype(str) if col != "label" else latent_df[col].astype(int)
            clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            clf.fit(x, y)
            probes[f"{name}_probe_accuracy"] = float(clf.score(x, y))
    return probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_splits", default="data/source_splits")
    parser.add_argument("--external_test", default="data/test/all_samples_prepared.csv")
    parser.add_argument("--full_features", default="features_by_dataset/combined_public_full_allfeatures/all_features.csv")
    parser.add_argument("--external_features", default="features_external/all_samples_full_allfeatures/all_features.csv")
    parser.add_argument("--train_sources", nargs="+", default=["leave_out_ghostbuster", "m4", "combined_strict"])
    parser.add_argument("--models", nargs="+", default=["qwen25_1_5b", "qwen25_7b"])
    parser.add_argument("--latent_dims", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--max_seq_len", nargs="+", type=int, default=[256])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--loss_configs", nargs="+", default=["A"])
    parser.add_argument("--output_dir", default="results_deep_dmd")
    parser.add_argument("--checkpoint_dir", default="checkpoints_deep_dmd")
    parser.add_argument("--feature_dir", default="features_deep_dmd")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--extract_features", action="store_true")
    parser.add_argument("--run_downstream_combos", action="store_true")
    parser.add_argument("--run_low_fpr_analysis", action="store_true")
    parser.add_argument("--run_probe", action="store_true")
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_recon", type=float, default=0.2)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--lambda_stability", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    ckpt_root = ROOT / args.checkpoint_dir
    feat_root = ROOT / args.feature_dir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    feat_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = build_base_datasets(ROOT / args.source_splits, ROOT / args.external_test, args.max_rows_per_split, args.seed)
    missing = [str(token_cache_path(m, n)) for m in args.models for n in base_names() if not token_cache_path(m, n).exists()]
    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "cuda": torch.cuda.is_available(), "missing_caches": missing, "args": vars(args)}, indent=2))
        return
    if missing:
        print("WARNING missing caches:", json.dumps(missing, indent=2))

    device = get_device(args.device)
    train_features = pd.read_csv(ROOT / args.full_features)
    ext_features = pd.read_csv(ROOT / args.external_features)
    full_cols = cleaned_full_columns(train_features)
    summary_rows = []
    arch_rows = []
    low_rows = []
    probe_rows = []
    error_rows = []
    best_by_train_model = {}
    comparison_rows = [{
        "model_version": "previous_best_transition",
        "train_source": "leave_out_ghostbuster",
        "feature_set": "full_plus_1_5b_and_7b_transition",
        "auroc": 0.6951,
        "auprc": 0.6592,
        "f1": 0.6799,
        "tpr_at_fpr_1pct": 0.0200,
        "tpr_at_fpr_5pct": 0.0933,
        "fpr_at_tpr_95pct": np.nan,
        "ece": 0.1488,
        "brier": 0.2459,
        "mcc": np.nan,
    }]

    for train_name in args.train_sources:
        train_meta, dev_meta = composite_train_dev(data, train_name)
        train_full = merge_full_features(train_meta, train_features, ext_features, full_cols)
        dev_full = merge_full_features(dev_meta, train_features, ext_features, full_cols)
        train_trans = merge_transition(train_full, train_name)
        dev_trans = merge_transition(dev_full, train_name)
        trans_cols = [c for c in train_trans.columns if c.endswith("_one5") or c.endswith("_seven")]
        for model_name in args.models:
            best = None
            for max_seq_len in args.max_seq_len:
                datasets, scaler, bins = build_deep_datasets(data, train_name, model_name, max_seq_len, min_tokens=20)
                for latent_dim in args.latent_dims:
                    for hidden_dim in args.hidden_dims:
                        for lr in args.learning_rates:
                            for loss_config_name in args.loss_configs:
                                loss_weights = LOSS_CONFIGS.get(loss_config_name, LOSS_CONFIGS["A"]).copy()
                                exp = f"{train_name}_{model_name}_seq{max_seq_len}_ld{latent_dim}_hd{hidden_dim}_lr{lr_tag(lr)}_cfg{loss_config_name}"
                                ckpt = ckpt_root / exp
                                model = DeepDMDEncoder(datasets["train"].input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
                                try:
                                    loaded = args.resume and load_if_available(model, ckpt, device)
                                    hist = pd.read_csv(ckpt / "training_history.csv") if (ckpt / "training_history.csv").exists() else pd.DataFrame()
                                    meta = {}
                                    if args.run_train and not loaded:
                                        model, hist, meta = train_deep_dmd(
                                            model,
                                            datasets["train"],
                                            datasets["dev"],
                                            output_dir=ckpt,
                                            device=device,
                                            epochs=args.epochs,
                                            batch_size=args.batch_size,
                                            lr=lr,
                                            patience=args.patience,
                                            seed=args.seed,
                                            loss_weights=loss_weights,
                                        )
                                        save_json(
                                            {"scaler": scaler.to_dict(), "loss_bins": bins, "args": vars(args), "model_name": model_name, "train_name": train_name, "max_seq_len": max_seq_len, "learning_rate": lr, "loss_config": loss_config_name, "loss_weights": loss_weights},
                                            ckpt / "preprocess.json",
                                        )
                                    dev_m = evaluate_deep_dmd(model, datasets["dev"], batch_size=args.batch_size, device=device)
                                    train_m = evaluate_deep_dmd(model, datasets["train"], batch_size=args.batch_size, device=device)
                                    last = hist.iloc[-1].to_dict() if not hist.empty else {}
                                    row = {
                                        "train_name": train_name,
                                        "model_name": model_name,
                                        "experiment": exp,
                                        "latent_dim": latent_dim,
                                        "hidden_dim": hidden_dim,
                                        "max_seq_len": max_seq_len,
                                        "learning_rate": lr,
                                        "loss_config": loss_config_name,
                                        "loaded_from_checkpoint": bool(loaded),
                                        "best_epoch": meta.get("best_epoch", np.nan),
                                        "early_stop_epoch": meta.get("early_stopping_epoch", np.nan),
                                        **{f"train_{k}": v for k, v in train_m.items()},
                                        **{f"dev_{k}": v for k, v in dev_m.items()},
                                        "last_train_loss": last.get("loss", np.nan),
                                        "last_pred_loss": last.get("pred_loss", np.nan),
                                        "last_cls_loss": last.get("cls_loss", np.nan),
                                        "last_spectral_radius": last.get("spectral_radius", np.nan),
                                    }
                                    arch_rows.append(row)
                                    rank = rank_tuple(dev_m)
                                    if best is None or rank > best[0]:
                                        best = (rank, exp, model, datasets, scaler, bins, latent_dim, hidden_dim, dev_m, max_seq_len, lr, loss_config_name)
                                except Exception as exc:
                                    error_rows.append({"created_at": now(), "train_name": train_name, "model_name": model_name, "experiment": exp, "error": repr(exc), "traceback": traceback.format_exc()})
                                    write_csv(pd.DataFrame(error_rows), out_dir / "error_log.csv")
            if best is None:
                continue
            _, exp, model, datasets, scaler, bins, latent_dim, hidden_dim, dev_m, max_seq_len, best_lr, best_loss_config = best
            best_by_train_model[(train_name, model_name)] = {"exp": exp, "model": model, "datasets": datasets, "latent_dim": latent_dim, "hidden_dim": hidden_dim, "max_seq_len": max_seq_len, "learning_rate": best_lr, "loss_config": best_loss_config}
            feature_dir = feat_root / exp
            feature_dir.mkdir(parents=True, exist_ok=True)
            pred_dev = predict_scores(model, datasets["dev"], batch_size=args.batch_size, device=device)
            for test_name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
                pred = predict_scores(model, datasets[test_name], batch_size=args.batch_size, device=device)
                m = save_score_eval(pred, out_dir / f"{exp}_deep_dmd_score_only_to_{test_name}")
                summary_rows.append({"train_name": train_name, "model_name": model_name, "experiment": exp, "feature_set": "deep_dmd_score_only", "test_set": test_name, "latent_dim": latent_dim, "hidden_dim": hidden_dim, **m})
            if args.extract_features:
                feature_paths = {}
                for ds_name, ds in datasets.items():
                    out = feature_dir / f"{ds_name}_deep_dmd_features.csv"
                    extract_deep_dmd_features(model, ds, out, batch_size=args.batch_size, device=device, prefix=f"deep_dmd_{model_name}")
                    feature_paths[ds_name] = out
                latent_train = pd.read_csv(str(feature_paths["train"]).replace("_deep_dmd_features.csv", "_deep_dmd_latent_pooled.csv"))
                probe = plot_latent_probe(latent_train, plot_dir, exp) if args.run_probe else {}
                probe_rows.append({"train_name": train_name, "model_name": model_name, "experiment": exp, **probe})
                deep_cols = [c for c in pd.read_csv(feature_paths["train"], nrows=1).columns if c != "id"]
                best_by_train_model[(train_name, model_name)]["feature_paths"] = feature_paths
                best_by_train_model[(train_name, model_name)]["deep_cols"] = deep_cols
                train_deep = train_full.merge(pd.read_csv(feature_paths["train"]), on="id", how="left")
                dev_deep = dev_full.merge(pd.read_csv(feature_paths["dev"]), on="id", how="left")
                train_deep_trans = train_trans.merge(pd.read_csv(feature_paths["train"]), on="id", how="left")
                dev_deep_trans = dev_trans.merge(pd.read_csv(feature_paths["dev"]), on="id", how="left")
                feature_sets = {
                    "deep_dmd_spectral_features_only": (train_deep, dev_deep, deep_cols),
                    "full_plus_deep_dmd_spectral": (train_deep, dev_deep, full_cols + deep_cols),
                    "full_plus_transition_plus_deep_dmd": (train_deep_trans, dev_deep_trans, full_cols + trans_cols + deep_cols),
                }
                for fs, (tr, dv, cols) in feature_sets.items():
                    if not args.run_downstream_combos:
                        continue
                    cols = [c for c in cols if c in tr.columns]
                    best_name, clf, med, _ = fit_select(tr, dv, cols)
                    joblib.dump(clf, ckpt_root / exp / f"{fs}_classifier.joblib")
                    for test_name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
                        test_base = merge_full_features(data[test_name], train_features, ext_features, full_cols)
                        if "transition" in fs:
                            test_df = merge_transition(test_base, train_name)
                        else:
                            test_df = test_base
                        ds_key = test_name
                        test_df = test_df.merge(pd.read_csv(feature_paths[ds_key]), on="id", how="left")
                        m = eval_one(clf, test_df, cols, med, out_dir / f"{exp}_{fs}_to_{test_name}")
                        summary_rows.append({"train_name": train_name, "model_name": model_name, "experiment": exp, "feature_set": fs, "test_set": test_name, "best_model": best_name, "latent_dim": latent_dim, "hidden_dim": hidden_dim, **m})
                # score feature and fusion
                if not args.run_downstream_combos:
                    continue
                train_score = train_full.merge(predict_scores(model, datasets["train"], batch_size=args.batch_size, device=device)[["id", "deep_dmd_score"]], on="id", how="left")
                dev_score = dev_full.merge(pred_dev[["id", "deep_dmd_score"]], on="id", how="left")
                cols = full_cols + ["deep_dmd_score"]
                best_name, clf, med, _ = fit_select(train_score, dev_score, cols)
                for test_name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
                    test_base = merge_full_features(data[test_name], train_features, ext_features, full_cols)
                    test_pred = predict_scores(model, datasets[test_name], batch_size=args.batch_size, device=device)
                    test_df = test_base.merge(test_pred[["id", "deep_dmd_score"]], on="id", how="left")
                    m = eval_one(clf, test_df, cols, med, out_dir / f"{exp}_full_plus_deep_dmd_score_to_{test_name}")
                    summary_rows.append({"train_name": train_name, "model_name": model_name, "experiment": exp, "feature_set": "full_plus_deep_dmd_score", "test_set": test_name, "best_model": best_name, "latent_dim": latent_dim, "hidden_dim": hidden_dim, **m})

                # Fusion: train full+transition model, choose alpha on dev.
                ft_cols = full_cols + trans_cols
                ft_name, ft_clf, ft_med, _ = fit_select(train_trans, dev_trans, ft_cols)
                x_dev, y_dev, _ = make_xy(dev_trans, ft_cols, ft_med)
                ft_dev = probabilities(ft_clf, x_dev)
                dev_fusion = dev_trans[["id", "label"]].copy()
                dev_fusion["transition_score"] = ft_dev
                dev_fusion = dev_fusion.merge(pred_dev[["id", "deep_dmd_score"]], on="id", how="inner")
                alpha, alpha_rows = choose_alpha(
                    dev_fusion["label"].to_numpy(dtype=int),
                    dev_fusion["transition_score"].to_numpy(dtype=float),
                    dev_fusion["deep_dmd_score"].to_numpy(dtype=float),
                )
                for test_name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
                    test_base = merge_full_features(data[test_name], train_features, ext_features, full_cols)
                    test_trans = merge_transition(test_base, train_name)
                    x_test, y_test, _ = make_xy(test_trans, ft_cols, ft_med)
                    ft_score = probabilities(ft_clf, x_test)
                    test_scores = test_trans[["id", "label"]].copy()
                    test_scores["transition_score"] = ft_score
                    test_scores = test_scores.merge(
                        predict_scores(model, datasets[test_name], batch_size=args.batch_size, device=device)[["id", "deep_dmd_score"]],
                        on="id",
                        how="inner",
                    )
                    score = alpha * test_scores["transition_score"].to_numpy(dtype=float) + (1 - alpha) * test_scores["deep_dmd_score"].to_numpy(dtype=float)
                    y_eval = test_scores["label"].to_numpy(dtype=int)
                    pred = pd.DataFrame({"id": test_scores["id"], "label": y_eval, "deep_dmd_score": score})
                    m = save_score_eval(pred, out_dir / f"{exp}_ensemble_fusion_to_{test_name}")
                    summary_rows.append({"train_name": train_name, "model_name": model_name, "experiment": exp, "feature_set": "ensemble_fusion", "test_set": test_name, "best_model": f"alpha={alpha:.2f}", "latent_dim": latent_dim, "hidden_dim": hidden_dim, **m})
                    if test_name == "all_samples" and args.run_low_fpr_analysis:
                        for target in [0.01, 0.05]:
                            dev_score = alpha * dev_fusion["transition_score"].to_numpy(dtype=float) + (1 - alpha) * dev_fusion["deep_dmd_score"].to_numpy(dtype=float)
                            thr, dev_tpr, dev_fpr = threshold_for_fpr(dev_fusion["label"].to_numpy(dtype=int), dev_score, target)
                            tm = detector_metrics(y_eval, score, y_pred=(score >= thr).astype(int), threshold=thr)
                            low_rows.append({"train_name": train_name, "model_name": model_name, "experiment": exp, "feature_set": "ensemble_fusion", "target_dev_fpr": target, "threshold": thr, "dev_tpr": dev_tpr, "dev_fpr": dev_fpr, "all_samples_tpr": tm["tpr"], "all_samples_fpr": tm["fpr"], "external_fpr_blowup_ratio": tm["fpr"] / max(dev_fpr, 1e-12), "all_samples_precision": tm["precision"], "all_samples_recall": tm["recall"], "all_samples_f1": tm["f1"]})

    # Cross-scale 1.5B+7B combinations are built only from public-dev-selected
    # best checkpoints for each scale. all_samples remains external-only.
    if args.run_downstream_combos and {"qwen25_1_5b", "qwen25_7b"}.issubset(set(args.models)):
        for train_name in args.train_sources:
            a = best_by_train_model.get((train_name, "qwen25_1_5b"))
            b = best_by_train_model.get((train_name, "qwen25_7b"))
            if not a or not b or "feature_paths" not in a or "feature_paths" not in b:
                continue
            train_meta, dev_meta = composite_train_dev(data, train_name)
            train_full = merge_full_features(train_meta, train_features, ext_features, full_cols)
            dev_full = merge_full_features(dev_meta, train_features, ext_features, full_cols)
            train_trans = merge_transition(train_full, train_name)
            dev_trans = merge_transition(dev_full, train_name)
            trans_cols = [c for c in train_trans.columns if c.endswith("_one5") or c.endswith("_seven")]
            pred_dev_15 = predict_scores(a["model"], a["datasets"]["dev"], batch_size=args.batch_size, device=device)
            pred_dev_7 = predict_scores(b["model"], b["datasets"]["dev"], batch_size=args.batch_size, device=device)
            dev_scores = dev_meta[["id", "label"]].merge(pred_dev_15[["id", "deep_dmd_score"]].rename(columns={"deep_dmd_score": "score_15"}), on="id", how="inner").merge(pred_dev_7[["id", "deep_dmd_score"]].rename(columns={"deep_dmd_score": "score_7"}), on="id", how="inner")
            alpha, _ = choose_alpha(dev_scores["label"].to_numpy(dtype=int), dev_scores["score_15"].to_numpy(dtype=float), dev_scores["score_7"].to_numpy(dtype=float))
            for test_name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
                p15 = predict_scores(a["model"], a["datasets"][test_name], batch_size=args.batch_size, device=device)
                p7 = predict_scores(b["model"], b["datasets"][test_name], batch_size=args.batch_size, device=device)
                scores = data[test_name][["id", "label"]].merge(p15[["id", "deep_dmd_score"]].rename(columns={"deep_dmd_score": "score_15"}), on="id", how="inner").merge(p7[["id", "deep_dmd_score"]].rename(columns={"deep_dmd_score": "score_7"}), on="id", how="inner")
                for fs, score in [
                    ("deep_dmd_score_average_1_5b_7b", 0.5 * scores["score_15"].to_numpy(dtype=float) + 0.5 * scores["score_7"].to_numpy(dtype=float)),
                    ("deep_dmd_score_weighted_fusion_1_5b_7b", alpha * scores["score_15"].to_numpy(dtype=float) + (1 - alpha) * scores["score_7"].to_numpy(dtype=float)),
                ]:
                    pred = pd.DataFrame({"id": scores["id"], "label": scores["label"], "deep_dmd_score": score})
                    m = save_score_eval(pred, out_dir / f"{train_name}_{fs}_to_{test_name}")
                    summary_rows.append({"train_name": train_name, "model_name": "qwen25_1_5b+qwen25_7b", "experiment": f"{a['exp']}__{b['exp']}", "feature_set": fs, "test_set": test_name, "best_model": f"alpha={alpha:.2f}" if "weighted" in fs else "average", **m})
            def merge_two_deep(meta_df: pd.DataFrame, key: str) -> pd.DataFrame:
                return meta_df.merge(pd.read_csv(a["feature_paths"][key]), on="id", how="left").merge(pd.read_csv(b["feature_paths"][key]), on="id", how="left")
            train_both = merge_two_deep(train_full, "train")
            dev_both = merge_two_deep(dev_full, "dev")
            train_both_trans = merge_two_deep(train_trans, "train")
            dev_both_trans = merge_two_deep(dev_trans, "dev")
            deep_cols = [c for c in train_both.columns if c.startswith("deep_dmd_")]
            combo_sets = {
                "deep_dmd_spectral_concat_1_5b_7b": (train_both, dev_both, deep_cols),
                "full_plus_deep_dmd_1_5b_7b": (train_both, dev_both, full_cols + deep_cols),
                "transition_plus_deep_dmd_1_5b_7b": (train_both_trans, dev_both_trans, trans_cols + deep_cols),
                "full_plus_transition_plus_deep_dmd_1_5b_7b": (train_both_trans, dev_both_trans, full_cols + trans_cols + deep_cols),
            }
            for fs, (tr, dv, cols) in combo_sets.items():
                cols = [c for c in cols if c in tr.columns]
                clf_name, clf, med, _ = fit_select(tr, dv, cols)
                for test_name in ["all_samples", "m4_test", "ghostbuster_test", "hc3_plus_test"]:
                    test_base = merge_full_features(data[test_name], train_features, ext_features, full_cols)
                    test_df = merge_transition(test_base, train_name) if "transition" in fs else test_base
                    test_df = test_df.merge(pd.read_csv(a["feature_paths"][test_name]), on="id", how="left").merge(pd.read_csv(b["feature_paths"][test_name]), on="id", how="left")
                    m = eval_one(clf, test_df, cols, med, out_dir / f"{train_name}_{fs}_to_{test_name}")
                    summary_rows.append({"train_name": train_name, "model_name": "qwen25_1_5b+qwen25_7b", "experiment": f"{a['exp']}__{b['exp']}", "feature_set": fs, "test_set": test_name, "best_model": clf_name, "n_features": len(cols), **m})

    summary = pd.DataFrame(summary_rows)
    write_csv(summary, out_dir / "deep_dmd_summary.csv")
    write_csv(summary, out_dir / "deep_dmd_full_sweep_summary.csv")
    write_csv(pd.DataFrame(arch_rows), out_dir / "architecture_search_results.csv")
    all_samples = summary[summary["test_set"].eq("all_samples")].sort_values("auroc", ascending=False)
    write_csv(all_samples, out_dir / "deep_dmd_all_samples_summary.csv")
    write_csv(pd.DataFrame(low_rows), out_dir / "low_fpr_deep_dmd_summary.csv")
    write_csv(pd.DataFrame(low_rows), out_dir / "low_fpr_transfer_summary.csv")
    write_csv(pd.DataFrame(probe_rows), out_dir / "probe_summary.csv")
    if not summary.empty:
        write_csv(summary[summary["feature_set"].astype(str).ne("deep_dmd_score_only")], out_dir / "downstream_feature_combo_summary.csv")
    if not all_samples.empty:
        for version, subset in [
            ("best_deep_dmd_score_only", all_samples[all_samples["feature_set"].eq("deep_dmd_score_only")]),
            ("best_deep_dmd_spectral", all_samples[all_samples["feature_set"].eq("deep_dmd_spectral_features_only")]),
            ("best_full_plus_deep_dmd", all_samples[all_samples["feature_set"].isin(["full_plus_deep_dmd_score", "full_plus_deep_dmd_spectral"])]),
            ("best_full_plus_transition_plus_deep_dmd", all_samples[all_samples["feature_set"].eq("full_plus_transition_plus_deep_dmd")]),
            ("best_fusion", all_samples[all_samples["feature_set"].eq("ensemble_fusion")]),
        ]:
            if subset.empty:
                continue
            r = subset.iloc[0]
            comparison_rows.append({"model_version": version, "train_source": r["train_name"], "feature_set": r["feature_set"], "auroc": r["auroc"], "auprc": r["auprc"], "f1": r["f1"], "tpr_at_fpr_1pct": r.get("tpr_at_fpr_1pct"), "tpr_at_fpr_5pct": r.get("tpr_at_fpr_5pct"), "fpr_at_tpr_95pct": r.get("fpr_at_tpr_95pct"), "ece": r.get("expected_calibration_error"), "brier": r.get("brier_score"), "mcc": r.get("mcc")})
    write_csv(pd.DataFrame(comparison_rows), out_dir / "deep_dmd_vs_previous_best.csv")
    if not all_samples.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        view = all_samples.head(20)[["train_name", "feature_set", "auroc", "auprc", "f1"]].melt(["train_name", "feature_set"], var_name="metric", value_name="value")
        view["model"] = view["train_name"] + "/" + view["feature_set"]
        sns.barplot(data=view, x="model", y="value", hue="metric", ax=ax)
        ax.tick_params(axis="x", rotation=40)
        ax.set_title("Deep DMD all_samples comparison")
        save_fig(fig, plot_dir / "deep_dmd_all_samples_performance")
    save_json({"created_at": now(), "args": vars(args), "device": str(device), "n_architecture_rows": len(arch_rows), "n_summary_rows": len(summary_rows), "errors": error_rows, "note": "all_samples used only as external test"}, out_dir / "deep_dmd_manifest.json")
    save_json({"created_at": now(), "args": vars(args), "device": str(device), "n_architecture_rows": len(arch_rows), "n_summary_rows": len(summary_rows), "errors": error_rows, "note": "all_samples used only as external test"}, out_dir / "deep_dmd_full_sweep_manifest.json")
    print(f"Wrote {out_dir / 'deep_dmd_summary.csv'}")


if __name__ == "__main__":
    main()
