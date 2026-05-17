"""Training for strict Text-Koopman lifting.

The training objective is unsupervised with respect to labels. Early stopping
uses dev reconstruction/DMD loss only, never all_samples or label metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .hidden_state_cache import load_hidden_cache_map
from .text_koopman_model import HiddenProjector, TextKoopmanLifting


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_loss_curves(history: list[dict] | pd.DataFrame, output_dir: str | Path) -> None:
    """Save train/dev loss curves as CSV-backed PNG/PDF artifacts."""
    df = pd.DataFrame(history)
    if df.empty or "epoch" not in df.columns:
        return
    try:
        import matplotlib.pyplot as plt

        out = Path(output_dir)
        pairs = [
            ("total", "Total loss"),
            ("recon", "Reconstruction loss"),
            ("dmd", "Local DMD loss"),
            ("multistep", "Multi-step DMD loss"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (key, title) in zip(axes.ravel(), pairs):
            train_col = f"train_{key}"
            dev_col = f"dev_{key}"
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col], label=train_col)
            if dev_col in df.columns:
                ax.plot(df["epoch"], df[dev_col], label=dev_col)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_loss_curves.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "training_loss_curves.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # The CSV remains the source of truth if plotting is unavailable.
        return


class HiddenTrajectoryDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        cache_map: dict[str, dict],
        projector: HiddenProjector,
        *,
        max_seq_len: int = 256,
        min_tokens: int = 20,
        max_rows: int | None = None,
        seed: int = 42,
    ) -> None:
        self.rows = []
        for _, row in meta.iterrows():
            rid = str(row["id"])
            rec = cache_map.get(rid)
            if rec is None:
                continue
            h = rec["hidden_states"]
            if h.shape[0] < min_tokens:
                continue
            if h.shape[0] > max_seq_len:
                h = h[:max_seq_len]
            # Keep the raw hidden trajectory in CPU cache and apply the
            # train-only projection lazily on the target device. Full-scale
            # runs otherwise spend most of their time doing CPU projection
            # during dataset construction.
            self.rows.append((rid, h.contiguous()))
        if max_rows is not None and len(self.rows) > max_rows:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(self.rows), size=int(max_rows), replace=False))
            self.rows = [self.rows[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rid, arr = self.rows[idx]
        return rid, arr.detach().clone().to(dtype=torch.float32)


def collate_trajectories(batch):
    return batch


def fit_projector_from_cache(
    train_meta: pd.DataFrame,
    cache_map: dict[str, dict],
    *,
    hidden_size: int,
    projection_dim: int,
    seed: int,
    projector_type: str = "random",
    max_tokens: int = 200_000,
) -> HiddenProjector:
    arrays = []
    n = 0
    for rid in train_meta["id"].astype(str):
        rec = cache_map.get(rid)
        if rec is not None:
            arr = rec["hidden_states"]
            if arr.numel() == 0:
                continue
            remaining = max_tokens - n
            if remaining <= 0:
                break
            if arr.shape[0] > remaining:
                idx = np.linspace(0, arr.shape[0] - 1, remaining).astype(int)
                arr = arr[idx]
            arrays.append(arr.float().numpy())
            n += int(arr.shape[0])
    return HiddenProjector(hidden_size=hidden_size, projection_dim=projection_dim, seed=seed, projector_type=projector_type).fit_scaler(arrays, max_tokens=max_tokens)


def _variance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    std = torch.std(z, dim=0, unbiased=False)
    return torch.mean(F.relu(0.1 - std) ** 2)


def _stability_loss(k: torch.Tensor) -> torch.Tensor:
    try:
        eig = torch.linalg.eigvals(k)
        return torch.mean(F.relu(torch.abs(eig) - 1.2) ** 2).real
    except Exception:
        return torch.tensor(0.0, device=k.device)


def _local_dmd_ridge(z: torch.Tensor, ridge: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cheap differentiable local DMD for training loss.

    Feature extraction still uses truncated SVD DMD for spectral fingerprints.
    Training only needs a stable per-document prediction objective, so using a
    small ridge solve on ``dmd_train_dim`` avoids slow SVD fallback on full runs.
    """
    x = z[:-1].T.contiguous()
    y = z[1:].T.contiguous()
    dim = x.shape[0]
    gram = x @ x.T + ridge * torch.eye(dim, dtype=x.dtype, device=x.device)
    rhs = y @ x.T
    k = torch.linalg.solve(gram.T, rhs.T).T
    k = torch.clamp(k, min=-10.0, max=10.0)
    return k, x, y


def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.zeros((), dtype=x.dtype, device=x.device)


def projector_tensors(projector: HiddenProjector, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
    mean = torch.as_tensor(projector.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(projector.std, dtype=torch.float32, device=device)
    proj = None if projector.projector_type == "linear" else torch.as_tensor(projector.projection, dtype=torch.float32, device=device)
    return mean, std, proj, projector.projector_type


def project_hidden(hidden: torch.Tensor, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str]) -> torch.Tensor:
    mean, std, proj, projector_type = tensors
    x = (hidden.to(mean.device, dtype=torch.float32) - mean) / std
    if projector_type == "linear":
        return x
    assert proj is not None
    return x @ proj


def batch_loss(
    model: TextKoopmanLifting,
    projector_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str],
    batch,
    *,
    dmd_rank: int,
    dmd_train_dim: int | None,
    lambda_recon: float,
    lambda_dmd: float,
    lambda_multistep: float,
    lambda_stability: float,
    lambda_var: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=device)
    parts = {"recon": 0.0, "dmd": 0.0, "multistep": 0.0, "stability": 0.0, "var": 0.0}
    n = 0
    for _, h_cpu in batch:
        x = project_hidden(h_cpu, projector_state)
        z, recon = model(x)
        recon_loss = F.mse_loss(recon, x)
        if not torch.isfinite(recon_loss).all():
            continue
        if dmd_train_dim is not None and z.shape[1] > dmd_train_dim:
            z_for_dmd = z[:, :dmd_train_dim]
        else:
            z_for_dmd = z
        try:
            k, x_r, y_r = _local_dmd_ridge(z_for_dmd)
            one_denom = torch.mean(y_r**2).clamp_min(1e-6)
            one = F.mse_loss(k @ x_r, y_r) / one_denom
            multi = torch.tensor(0.0, device=device)
            steps = 0
            for step in [2, 4, 8]:
                if x_r.shape[1] > step:
                    pred = torch.linalg.matrix_power(k, step) @ x_r[:, :-step]
                    target = x_r[:, step:]
                    multi = multi + F.mse_loss(pred, target) / torch.mean(target**2).clamp_min(1e-6)
                    steps += 1
            if steps:
                multi = multi / steps
            stab = _stability_loss(k)
        except Exception:
            one = torch.tensor(0.0, device=device)
            multi = torch.tensor(0.0, device=device)
            stab = torch.tensor(0.0, device=device)
        var = _variance_loss(z)
        one = _finite_or_zero(one)
        multi = _finite_or_zero(multi)
        stab = _finite_or_zero(stab)
        var = _finite_or_zero(var)
        one = torch.clamp(one, max=10.0)
        multi = torch.clamp(multi, max=10.0)
        stab = torch.clamp(stab, max=10.0)
        sample_loss = lambda_recon * recon_loss + lambda_dmd * one + lambda_multistep * multi + lambda_stability * stab + lambda_var * var
        if not torch.isfinite(sample_loss).all():
            sample_loss = lambda_recon * recon_loss + lambda_var * var
        if not torch.isfinite(sample_loss).all():
            continue
        total = total + sample_loss
        parts["recon"] += float(recon_loss.detach().cpu())
        parts["dmd"] += float(one.detach().cpu())
        parts["multistep"] += float(multi.detach().cpu())
        parts["stability"] += float(stab.detach().cpu())
        parts["var"] += float(var.detach().cpu())
        n += 1
    if n:
        total = total / n
        parts = {k: v / n for k, v in parts.items()}
    parts["total"] = float(total.detach().cpu())
    return total, parts


def evaluate_loss(model: TextKoopmanLifting, projector_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str], loader: DataLoader, device: torch.device, **loss_kwargs) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            _, parts = batch_loss(model, projector_state, batch, device=device, **loss_kwargs)
            rows.append(parts)
    if not rows:
        return {"total": np.inf}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def train_text_koopman_lifting(
    *,
    train_meta: pd.DataFrame,
    dev_meta: pd.DataFrame,
    hidden_root: str | Path,
    model_name: str,
    dataset_names: list[str],
    output_dir: str | Path,
    projection_dim: int = 128,
    projector_type: str = "random",
    observable_dim: int = 512,
    hidden_dim: int = 512,
    dropout: float = 0.0,
    dmd_rank: int = 32,
    dmd_train_dim: int | None = None,
    max_seq_len: int = 256,
    min_tokens: int = 20,
    max_train_sequences: int | None = None,
    max_dev_sequences: int | None = None,
    max_effective_epochs: int | None = None,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | torch.device = "auto",
    lambda_recon: float = 0.5,
    lambda_dmd: float = 1.0,
    lambda_multistep: float = 0.5,
    lambda_stability: float = 0.1,
    lambda_var: float = 0.01,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    cache_map = load_hidden_cache_map(hidden_root, model_name, dataset_names)
    if not cache_map:
        raise FileNotFoundError(f"No hidden-state cache found for {model_name} in {hidden_root}")
    first = next(iter(cache_map.values()))
    hidden_size = int(first.get("hidden_size", first["hidden_states"].shape[1]))
    projector = fit_projector_from_cache(train_meta, cache_map, hidden_size=hidden_size, projection_dim=projection_dim, seed=seed, projector_type=projector_type)
    train_ds = HiddenTrajectoryDataset(train_meta, cache_map, projector, max_seq_len=max_seq_len, min_tokens=min_tokens, max_rows=max_train_sequences, seed=seed)
    dev_ds = HiddenTrajectoryDataset(dev_meta, cache_map, projector, max_seq_len=max_seq_len, min_tokens=min_tokens, max_rows=max_dev_sequences, seed=seed + 1)
    if not train_ds or not dev_ds:
        raise ValueError(f"Empty train/dev hidden trajectory dataset: train={len(train_ds)} dev={len(dev_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectories)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectories)
    model_input_dim = hidden_size if projector_type == "linear" else projection_dim
    model = TextKoopmanLifting(projection_dim=projection_dim, observable_dim=observable_dim, hidden_dim=hidden_dim, dropout=dropout, input_dim=model_input_dim).to(device)
    proj_state = projector_tensors(projector, device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_kwargs = {
        "dmd_rank": dmd_rank,
        "dmd_train_dim": int(dmd_train_dim or min(max(dmd_rank * 2, 32), observable_dim)),
        "lambda_recon": lambda_recon,
        "lambda_dmd": lambda_dmd,
        "lambda_multistep": lambda_multistep,
        "lambda_stability": lambda_stability,
        "lambda_var": lambda_var,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = (np.inf, -1, None)
    bad = 0
    history = []
    requested_epochs = int(epochs)
    if max_effective_epochs is not None:
        epochs = min(int(epochs), int(max_effective_epochs))
    for epoch in range(1, epochs + 1):
        model.train()
        train_parts = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, parts = batch_loss(model, proj_state, batch, device=device, **loss_kwargs)
            if not loss.requires_grad:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_parts.append(parts)
        if not train_parts:
            raise RuntimeError("All training batches were skipped due to non-finite losses.")
        train_mean = {k: float(np.mean([p[k] for p in train_parts])) for k in train_parts[0]}
        dev_mean = evaluate_loss(model, proj_state, dev_loader, device, **loss_kwargs)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_mean.items()}, **{f"dev_{k}": v for k, v in dev_mean.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "training_history_partial.csv", index=False)
        if dev_mean["total"] < best[0]:
            best = (dev_mean["total"], epoch, {k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best[2] is not None:
        model.load_state_dict(best[2])
    torch.save({"model_state": model.state_dict()}, out / "text_koopman_lifting.pt")
    joblib.dump(projector, out / "projector.joblib")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    save_loss_curves(history, out)
    meta = {
        "created_at": now(),
        "model_name": model_name,
        "hidden_size": hidden_size,
        "projection_dim": int(projection_dim),
        "projector_type": projector_type,
        "model_input_dim": int(model_input_dim),
        "projector_pca_explained_variance_sum": None
        if getattr(projector, "explained_variance_ratio", None) is None
        else float(np.sum(projector.explained_variance_ratio)),
        "observable_dim": int(observable_dim),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "dmd_rank": int(dmd_rank),
        "dmd_train_dim": int(loss_kwargs["dmd_train_dim"]),
        "max_seq_len": int(max_seq_len),
        "max_train_sequences": None if max_train_sequences is None else int(max_train_sequences),
        "max_dev_sequences": None if max_dev_sequences is None else int(max_dev_sequences),
        "requested_epochs": int(requested_epochs),
        "effective_epochs_cap": None if max_effective_epochs is None else int(max_effective_epochs),
        "n_train_sequences_used": int(len(train_ds)),
        "n_dev_sequences_used": int(len(dev_ds)),
        "best_epoch": int(best[1]),
        "early_stopping_epoch": int(history[-1]["epoch"]) if history else None,
        "selection": "dev unsupervised reconstruction/local-DMD loss; no labels and no all_samples",
        "uses_label_classifier": False,
        "uses_global_K_parameter": False,
        "classifier_input_policy": "not applicable; lifting checkpoint emits no classifier features",
        "loss_weights": {
            "lambda_recon": lambda_recon,
            "lambda_dmd": lambda_dmd,
            "lambda_multistep": lambda_multistep,
            "lambda_stability": lambda_stability,
            "lambda_var": lambda_var,
        },
    }
    (out / "lifting_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": model, "projector": projector, "metadata": meta, "history": history}
