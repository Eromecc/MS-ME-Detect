"""Per-document local Koopman/DMD spectral features.

No pooled hidden state, pooled latent state, token id, or text feature is
emitted. Rows contain scalar spectral/residual summaries from each document's
local DMD operator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .hidden_state_cache import iter_hidden_records


SPECTRAL_PREFIX = "text_koopman_"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def entropy(vals: np.ndarray, bins: int = 16) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    hist, _ = np.histogram(vals, bins=bins)
    p = hist.astype(float)
    p = p[p > 0] / max(float(p.sum()), 1.0)
    return float(-(p * np.log(p + 1e-12)).sum())


def effective_rank(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return np.nan
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def local_dmd_reduced(z: torch.Tensor, rank: int = 32, ridge: float = 1e-6):
    """Return reduced local K, reduced coordinates and one-step targets."""
    if z.ndim != 2 or z.shape[0] < 3:
        raise ValueError("z must have shape [seq_len, observable_dim] with seq_len >= 3")
    x = z[:-1].T.contiguous()
    y = z[1:].T.contiguous()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    r = int(min(rank, s.numel()))
    if r < 1:
        raise ValueError("rank too small")
    ur = u[:, :r]
    sr = torch.clamp(s[:r], min=ridge)
    vr = vh[:r, :].T
    k_tilde = ur.T @ y @ vr @ torch.diag(1.0 / sr)
    x_r = ur.T @ x
    y_r = ur.T @ y
    return k_tilde, x_r, y_r


def spectral_features_from_k(k: np.ndarray, prefix: str = SPECTRAL_PREFIX) -> dict[str, float]:
    eig = np.linalg.eigvals(k)
    abs_eig = np.abs(eig)
    angles = np.angle(eig)
    s = np.linalg.svd(k, compute_uv=False)
    energy = s**2
    total_energy = float(energy.sum()) if energy.size else 0.0
    real = np.real(eig)
    imag = np.imag(eig)
    return {
        f"{prefix}spectral_radius": float(np.max(abs_eig)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_mean": float(np.mean(abs_eig)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_std": float(np.std(abs_eig)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_min": float(np.min(abs_eig)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_max": float(np.max(abs_eig)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_median": float(np.median(abs_eig)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_q25": float(np.quantile(abs_eig, 0.25)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_q75": float(np.quantile(abs_eig, 0.75)) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_iqr": float(np.quantile(abs_eig, 0.75) - np.quantile(abs_eig, 0.25)) if abs_eig.size else np.nan,
        f"{prefix}stable_eig_ratio_abs_le_1": float(np.mean(abs_eig <= 1.0)) if abs_eig.size else np.nan,
        f"{prefix}unstable_eig_ratio_abs_gt_1": float(np.mean(abs_eig > 1.0)) if abs_eig.size else np.nan,
        f"{prefix}near_unit_circle_ratio_abs_0_9_1_1": float(np.mean((abs_eig >= 0.9) & (abs_eig <= 1.1))) if abs_eig.size else np.nan,
        f"{prefix}eig_abs_entropy": entropy(abs_eig),
        f"{prefix}complex_eig_ratio": float(np.mean(np.abs(imag) > 1e-8)) if eig.size else np.nan,
        f"{prefix}eig_angle_mean": float(np.mean(angles)) if angles.size else np.nan,
        f"{prefix}eig_angle_std": float(np.std(angles)) if angles.size else np.nan,
        f"{prefix}eig_angle_entropy": entropy(angles),
        f"{prefix}positive_real_ratio": float(np.mean(real > 1e-8)) if eig.size else np.nan,
        f"{prefix}negative_real_ratio": float(np.mean(real < -1e-8)) if eig.size else np.nan,
        f"{prefix}singular_value_top1": float(s[0]) if s.size else np.nan,
        f"{prefix}singular_value_top3_energy_ratio": float(energy[:3].sum() / total_energy) if total_energy else np.nan,
        f"{prefix}singular_value_top5_energy_ratio": float(energy[:5].sum() / total_energy) if total_energy else np.nan,
        f"{prefix}nuclear_norm": float(s.sum()) if s.size else np.nan,
        f"{prefix}fro_norm": float(np.linalg.norm(k, ord="fro")),
        f"{prefix}condition_number": float(np.linalg.cond(k)) if k.size else np.nan,
        f"{prefix}effective_rank": effective_rank(s),
        f"{prefix}low_rank_energy_ratio": float(energy[: min(5, len(energy))].sum() / total_energy) if total_energy else np.nan,
    }


def residual_features(k: torch.Tensor, x_r: torch.Tensor, y_r: torch.Tensor, prefix: str = SPECTRAL_PREFIX) -> dict[str, float]:
    y_hat = k @ x_r
    err = y_r - y_hat
    denom = torch.mean(y_r**2).clamp_min(1e-8)
    out = {
        f"{prefix}one_step_dmd_mse": float(torch.mean(err**2).detach().cpu()),
        f"{prefix}one_step_dmd_mae": float(torch.mean(torch.abs(err)).detach().cpu()),
        f"{prefix}normalized_dmd_error": float((torch.mean(err**2) / denom).detach().cpu()),
    }
    for step in [2, 4, 8]:
        if x_r.shape[1] <= step:
            out[f"{prefix}multistep_mse_k{step}"] = np.nan
            continue
        kp = torch.linalg.matrix_power(k, step)
        pred = kp @ x_r[:, :-step]
        target = x_r[:, step:]
        out[f"{prefix}multistep_mse_k{step}"] = float(torch.mean((pred - target) ** 2).detach().cpu())
    return out


def trajectory_features(z: torch.Tensor, prefix: str = SPECTRAL_PREFIX) -> dict[str, float]:
    vel = z[1:] - z[:-1]
    acc = vel[1:] - vel[:-1] if vel.shape[0] > 1 else torch.empty((0, z.shape[1]), device=z.device)
    zn = torch.linalg.norm(z, dim=1)
    vn = torch.linalg.norm(vel, dim=1) if vel.numel() else torch.tensor([], device=z.device)
    an = torch.linalg.norm(acc, dim=1) if acc.numel() else torch.tensor([], device=z.device)
    return {
        f"{prefix}observable_velocity_mean": float(vn.mean().detach().cpu()) if vn.numel() else np.nan,
        f"{prefix}observable_velocity_std": float(vn.std(unbiased=False).detach().cpu()) if vn.numel() else np.nan,
        f"{prefix}observable_acceleration_mean": float(an.mean().detach().cpu()) if an.numel() else np.nan,
        f"{prefix}observable_acceleration_std": float(an.std(unbiased=False).detach().cpu()) if an.numel() else np.nan,
        f"{prefix}observable_norm_mean": float(zn.mean().detach().cpu()) if zn.numel() else np.nan,
        f"{prefix}observable_norm_std": float(zn.std(unbiased=False).detach().cpu()) if zn.numel() else np.nan,
    }


def extract_text_koopman_features(
    *,
    hidden_root: str | Path,
    model_name: str,
    dataset_name: str,
    lifting_model,
    projector,
    output_csv: str | Path,
    dmd_rank: int = 32,
    min_tokens: int = 20,
    allowed_ids: set[str] | None = None,
    checkpoint_created_at: str | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    device = torch.device(device)
    lifting_model.to(device)
    lifting_model.eval()
    rows = []
    failed = []
    short = []
    with torch.no_grad():
        for record in iter_hidden_records(hidden_root, model_name, dataset_name):
            rid = str(record["id"])
            if allowed_ids is not None and rid not in allowed_ids:
                continue
            hidden = record["hidden_states"].float()
            if hidden.shape[0] < min_tokens:
                short.append(rid)
                continue
            try:
                projected = projector.transform_tensor(hidden, device=device)
                z, _ = lifting_model(projected)
                k, x_r, y_r = local_dmd_reduced(z, rank=dmd_rank)
                k_np = k.detach().cpu().numpy()
                row = {"id": rid}
                row.update(spectral_features_from_k(k_np))
                row.update(residual_features(k, x_r, y_r))
                row.update(trajectory_features(z))
                rows.append(row)
            except Exception:
                failed.append(rid)
    out = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.replace([np.inf, -np.inf], np.nan).to_csv(out_path, index=False)
    manifest = {
        "created_at": now(),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "output_csv": str(out_path),
        "n_rows": int(len(out)),
        "n_features": int(max(len(out.columns) - 1, 0)),
        "failed_ids": failed,
        "short_ids": short,
        "dmd_rank": int(dmd_rank),
        "allowed_ids_count": int(len(allowed_ids)) if allowed_ids is not None else None,
        "checkpoint_created_at": checkpoint_created_at,
        "feature_policy": "spectral/residual/trajectory scalar summaries only; no pooled hidden, pooled z, token ids, or text.",
    }
    out_path.with_name(out_path.stem + "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_feature_artifacts(checkpoint_dir: str | Path, device: str | torch.device = "cpu"):
    from .text_koopman_model import TextKoopmanLifting

    checkpoint_dir = Path(checkpoint_dir)
    projector = joblib.load(checkpoint_dir / "projector.joblib")
    meta = json.loads((checkpoint_dir / "lifting_metadata.json").read_text(encoding="utf-8"))
    model = TextKoopmanLifting(
        projection_dim=int(meta["projection_dim"]),
        observable_dim=int(meta["observable_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        dropout=float(meta.get("dropout", 0.0)),
        input_dim=int(meta.get("model_input_dim", meta["projection_dim"])),
    )
    payload = torch.load(checkpoint_dir / "text_koopman_lifting.pt", map_location=device)
    model.load_state_dict(payload["model_state"])
    return model, projector, meta
