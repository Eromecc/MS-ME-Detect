"""Training utilities for Deep DMD / Koopman encoder experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .deep_dmd_dataset import collate_deep_dmd
from .train_eval import detector_metrics, probabilities


def sequence_pair_mask(mask: torch.Tensor, step: int) -> torch.Tensor:
    return mask[:, :-step] & mask[:, step:]


def deep_dmd_loss(
    model,
    batch: dict,
    *,
    lambda_pred: float = 1.0,
    lambda_recon: float = 0.2,
    lambda_cls: float = 0.5,
    lambda_reg: float = 1e-4,
    lambda_stability: float = 0.1,
    steps: list[int] | None = None,
) -> tuple[torch.Tensor, dict]:
    steps = steps or [1, 2, 4, 8]
    x = batch["x"]
    mask = batch["mask"]
    y = batch["label"]
    out = model(x, mask)
    z = out["z"]
    total_pred = torch.tensor(0.0, device=x.device)
    pred_parts = {}
    for step in steps:
        if z.shape[1] <= step:
            continue
        pair = sequence_pair_mask(mask, step)
        if not pair.any():
            continue
        pred = model.apply_k(z[:, :-step], step)
        target = z[:, step:].detach()
        loss = ((pred - target) ** 2).mean(dim=-1)[pair].mean()
        total_pred = total_pred + loss
        pred_parts[f"pred_k{step}"] = float(loss.detach().cpu())
    recon_loss = torch.tensor(0.0, device=x.device)
    if out["recon"] is not None:
        recon_loss = ((out["recon"] - x) ** 2).mean(dim=-1)[mask].mean()
    cls_loss = F.binary_cross_entropy_with_logits(out["logits"], y)
    reg_loss = torch.linalg.matrix_norm(model.K, ord="fro") ** 2
    eig = torch.linalg.eigvals(model.K)
    spectral_radius = torch.abs(eig).max()
    stability = torch.relu(torch.abs(eig) - 1.2).pow(2).mean()
    loss = lambda_pred * total_pred + lambda_recon * recon_loss + lambda_cls * cls_loss + lambda_reg * reg_loss + lambda_stability * stability
    parts = {
        "loss": float(loss.detach().cpu()),
        "pred_loss": float(total_pred.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "stability_loss": float(stability.detach().cpu()),
        "spectral_radius": float(spectral_radius.detach().cpu()),
        **pred_parts,
    }
    return loss, parts


@torch.no_grad()
def predict_scores(model, dataset, *, batch_size: int, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_deep_dmd)
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask)["logits"]
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["label"].numpy().astype(int)
        for i, rid in enumerate(batch["id"]):
            row = {
                "id": rid,
                "label": int(labels[i]),
                "deep_dmd_score": float(prob[i]),
                "prediction": int(prob[i] >= 0.5),
            }
            row.update(batch["meta"][i])
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_deep_dmd(model, dataset, *, batch_size: int, device: torch.device) -> dict:
    pred = predict_scores(model, dataset, batch_size=batch_size, device=device)
    if pred.empty:
        return {}
    return detector_metrics(pred["label"].astype(int), pred["deep_dmd_score"], y_pred=pred["prediction"])


def train_deep_dmd(
    model,
    train_dataset,
    dev_dataset,
    *,
    output_dir: Path,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    loss_weights: dict | None = None,
) -> tuple[object, pd.DataFrame, dict]:
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_deep_dmd)
    history = []
    best_rank = None
    best_state = None
    best_epoch = -1
    stale = 0
    loss_weights = loss_weights or {}
    for epoch in range(1, epochs + 1):
        model.train()
        parts_acc = []
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, parts = deep_dmd_loss(model, batch, **loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            parts_acc.append(parts)
        train_metrics = evaluate_deep_dmd(model, train_dataset, batch_size=batch_size, device=device)
        dev_metrics = evaluate_deep_dmd(model, dev_dataset, batch_size=batch_size, device=device)
        avg = {k: float(np.mean([p[k] for p in parts_acc if k in p])) for k in sorted(set().union(*[p.keys() for p in parts_acc]))} if parts_acc else {}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"dev_{k}": v for k, v in dev_metrics.items()}, **avg}
        history.append(row)
        rank = tuple(float(dev_metrics.get(k, -np.inf)) for k in ["auprc", "auroc", "tpr_at_fpr_5pct", "f1"])
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist = pd.DataFrame(history)
    hist.to_csv(output_dir / "training_history.csv", index=False)
    meta = {"best_epoch": best_epoch, "early_stopping_epoch": int(hist["epoch"].max()) if not hist.empty else None, "selection": "dev auprc>auroc>tpr_at_fpr_5pct>f1"}
    (output_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "model_config": {"input_dim": model.input_dim, "latent_dim": model.latent_dim}}, output_dir / "deep_dmd_model.pt")
    return model, hist, meta
