"""Feature extraction from trained Deep DMD latent trajectories and K."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .deep_dmd_dataset import collate_deep_dmd
from .utils import write_csv


def entropy(values: np.ndarray, bins: int = 12, rng=None) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan
    counts, _ = np.histogram(vals, bins=bins, range=rng)
    if counts.sum() <= 0:
        return 0.0
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def global_k_features(k: np.ndarray, prefix: str = "deep_dmd") -> dict:
    eig = np.linalg.eigvals(k)
    abs_e = np.abs(eig)
    sing = np.linalg.svd(k, compute_uv=False)
    e2 = sing**2
    total = e2.sum() + 1e-12
    return {
        f"{prefix}_spectral_radius": float(abs_e.max()),
        f"{prefix}_eig_abs_mean": float(abs_e.mean()),
        f"{prefix}_eig_abs_std": float(abs_e.std()),
        f"{prefix}_eig_abs_min": float(abs_e.min()),
        f"{prefix}_eig_abs_max": float(abs_e.max()),
        f"{prefix}_stable_eig_ratio": float(np.mean(abs_e <= 1.0)),
        f"{prefix}_near_unit_circle_ratio": float(np.mean((abs_e >= 0.9) & (abs_e <= 1.1))),
        f"{prefix}_complex_eig_ratio": float(np.mean(np.abs(np.imag(eig)) > 1e-8)),
        f"{prefix}_eig_angle_entropy": entropy(np.angle(eig), rng=(-np.pi, np.pi)),
        f"{prefix}_K_fro_norm": float(np.linalg.norm(k, ord="fro")),
        f"{prefix}_K_trace": float(np.real(np.trace(k))),
        f"{prefix}_K_rank": float(np.linalg.matrix_rank(k)),
        f"{prefix}_K_condition_number": float(np.linalg.cond(k)),
        f"{prefix}_K_nuclear_norm": float(sing.sum()),
        f"{prefix}_top1_singular_value": float(sing[0]) if sing.size else np.nan,
        f"{prefix}_top3_singular_energy_ratio": float(e2[: min(3, len(e2))].sum() / total),
        f"{prefix}_low_rank_energy_ratio": float(e2[: min(2, len(e2))].sum() / total),
    }


def sample_latent_features(z: np.ndarray, x: np.ndarray, recon: np.ndarray | None, k: np.ndarray, prefix: str = "deep_dmd") -> dict:
    out = {}
    if len(z) < 3:
        return out
    pred1 = z[:-1] @ k.T
    err1 = z[1:] - pred1
    out[f"{prefix}_one_step_latent_prediction_mse"] = float(np.mean(err1**2))
    for step in [2, 4, 8]:
        if len(z) > step:
            kp = np.linalg.matrix_power(k, step)
            err = z[step:] - (z[:-step] @ kp.T)
            out[f"{prefix}_multistep_prediction_mse_k{step}"] = float(np.mean(err**2))
        else:
            out[f"{prefix}_multistep_prediction_mse_k{step}"] = np.nan
    vel = np.linalg.norm(np.diff(z, axis=0), axis=1)
    acc = np.linalg.norm(np.diff(z, n=2, axis=0), axis=1)
    norm = np.linalg.norm(z, axis=1)
    residual = np.linalg.norm(err1, axis=1)
    out.update(
        {
            f"{prefix}_latent_velocity_mean": float(vel.mean()),
            f"{prefix}_latent_velocity_std": float(vel.std()),
            f"{prefix}_latent_acceleration_mean": float(acc.mean()) if len(acc) else np.nan,
            f"{prefix}_latent_acceleration_std": float(acc.std()) if len(acc) else np.nan,
            f"{prefix}_latent_norm_mean": float(norm.mean()),
            f"{prefix}_latent_norm_std": float(norm.std()),
            f"{prefix}_koopman_residual_mean": float(residual.mean()),
            f"{prefix}_koopman_residual_std": float(residual.std()),
        }
    )
    if recon is not None:
        out[f"{prefix}_reconstruction_mse"] = float(np.mean((recon - x) ** 2))
    else:
        out[f"{prefix}_reconstruction_mse"] = np.nan
    return out


@torch.no_grad()
def extract_deep_dmd_features(model, dataset, output_path, *, batch_size: int, device, prefix: str = "deep_dmd") -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_deep_dmd)
    model.eval()
    k = model.K.detach().cpu().numpy()
    k_feats = global_k_features(k, prefix)
    rows = []
    latent_rows = []
    for batch in loader:
        x_t = batch["x"].to(device)
        mask_t = batch["mask"].to(device)
        out = model(x_t, mask_t)
        z = out["z"].detach().cpu().numpy()
        recon = out["recon"].detach().cpu().numpy() if out["recon"] is not None else None
        x_np = batch["x"].numpy()
        for i, rid in enumerate(batch["id"]):
            n = int(batch["mask"][i].sum().item())
            feats = {"id": rid, **k_feats}
            feats.update(sample_latent_features(z[i, :n], x_np[i, :n], recon[i, :n] if recon is not None else None, k, prefix))
            rows.append(feats)
            pooled = z[i, :n].mean(axis=0)
            meta = batch["meta"][i]
            latent_rows.append({"id": rid, "label": int(batch["label"][i].item()), **meta, **{f"latent_{j}": float(v) for j, v in enumerate(pooled)}})
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    write_csv(df, output_path)
    latent_path = str(output_path).replace("_deep_dmd_features.csv", "_deep_dmd_latent_pooled.csv")
    write_csv(pd.DataFrame(latent_rows), latent_path)
    return df
