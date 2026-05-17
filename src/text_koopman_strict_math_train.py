"""Training for strict mathematical Text-Koopman lifting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .hidden_state_cache import load_hidden_cache_map
from .text_koopman_strict_math_features import local_exact_dmd_reduced
from .text_koopman_strict_math_model import StrictKoopmanLifting


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictHiddenTrajectoryDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        cache_map: dict[str, dict],
        *,
        max_seq_len: int,
        min_tokens: int = 20,
        max_rows: int | None = None,
        seed: int = 42,
    ) -> None:
        rows = []
        for _, row in meta.iterrows():
            rid = str(row["id"])
            rec = cache_map.get(rid)
            if rec is None:
                continue
            h = rec["hidden_states"]
            if h.shape[0] > max_seq_len:
                h = h[:max_seq_len]
            if h.shape[0] < min_tokens:
                continue
            rows.append((rid, h.contiguous()))
        if max_rows is not None and len(rows) > max_rows:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(rows), size=int(max_rows), replace=False))
            rows = [rows[int(i)] for i in idx]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rid, h = self.rows[idx]
        return rid, h.detach().clone().to(dtype=torch.float32)


def collate_trajectories(batch):
    return batch


def save_loss_curves(history: list[dict], output_dir: str | Path) -> None:
    df = pd.DataFrame(history)
    if df.empty:
        return
    out = Path(output_dir)
    df.to_csv(out / "training_history.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, key in zip(axes.ravel(), ["total", "recon", "dmd", "multistep"]):
            for split in ["train", "dev"]:
                col = f"{split}_{key}"
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label=col)
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_loss_curves.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "training_loss_curves.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        return


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


def _finite(x: torch.Tensor) -> torch.Tensor:
    return x if torch.isfinite(x).all() else torch.zeros((), dtype=x.dtype, device=x.device)


def batch_loss(
    model: StrictKoopmanLifting,
    batch,
    *,
    dmd_rank: int,
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
        h = h_cpu.to(device, dtype=torch.float32)
        z, recon = model(h)
        recon_loss = F.mse_loss(recon, h)
        try:
            k, ur, x, y, x_r, _ = local_exact_dmd_reduced(z, rank=dmd_rank)
            y_hat = ur @ (k @ x_r)
            one = F.mse_loss(y_hat, y) / torch.mean(y**2).clamp_min(1e-6)
            multi = torch.tensor(0.0, device=device)
            steps = 0
            for step in [2, 4]:
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
        recon_loss = _finite(recon_loss)
        one = torch.clamp(_finite(one), max=10.0)
        multi = torch.clamp(_finite(multi), max=10.0)
        stab = torch.clamp(_finite(stab), max=10.0)
        var = _finite(var)
        loss = lambda_recon * recon_loss + lambda_dmd * one + lambda_multistep * multi + lambda_stability * stab + lambda_var * var
        if not torch.isfinite(loss).all():
            continue
        total = total + loss
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


def evaluate_loss(model: StrictKoopmanLifting, loader: DataLoader, device: torch.device, **loss_kwargs) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            _, parts = batch_loss(model, batch, device=device, **loss_kwargs)
            rows.append(parts)
    if not rows:
        return {"total": np.inf, "recon": np.inf, "dmd": np.inf}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cublas" in msg


def train_strict_math_lifting(
    *,
    train_meta: pd.DataFrame,
    dev_meta: pd.DataFrame,
    hidden_root: str | Path,
    model_name: str,
    dataset_names: list[str],
    output_dir: str | Path,
    observable_dim: int | None = None,
    observable_multiplier: int = 2,
    dmd_rank: int = 32,
    max_seq_len: int = 256,
    min_tokens: int = 20,
    max_train_sequences: int | None = None,
    max_dev_sequences: int | None = None,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 1,
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
    observable_dim = int(observable_dim or (observable_multiplier * hidden_size))
    if observable_dim <= hidden_size:
        raise ValueError(f"observable_dim={observable_dim} must be > hidden_size={hidden_size}")

    train_ds = StrictHiddenTrajectoryDataset(train_meta, cache_map, max_seq_len=max_seq_len, min_tokens=min_tokens, max_rows=max_train_sequences, seed=seed)
    dev_ds = StrictHiddenTrajectoryDataset(dev_meta, cache_map, max_seq_len=max_seq_len, min_tokens=min_tokens, max_rows=max_dev_sequences, seed=seed + 1)
    if not train_ds or not dev_ds:
        raise ValueError(f"Empty train/dev hidden trajectory dataset: train={len(train_ds)} dev={len(dev_ds)}")
    if batch_size != 1:
        # Keep memory behavior explicit for this strict high-dimensional run.
        batch_size = 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectories)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectories)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = StrictKoopmanLifting(hidden_size=hidden_size, observable_dim=observable_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_kwargs = {
        "dmd_rank": int(dmd_rank),
        "lambda_recon": lambda_recon,
        "lambda_dmd": lambda_dmd,
        "lambda_multistep": lambda_multistep,
        "lambda_stability": lambda_stability,
        "lambda_var": lambda_var,
    }
    best = (np.inf, -1, None)
    bad = 0
    history: list[dict] = []
    oom_events: list[dict] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_parts = []
        for batch in train_loader:
            try:
                opt.zero_grad(set_to_none=True)
                loss, parts = batch_loss(model, batch, device=device, **loss_kwargs)
                if not loss.requires_grad:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_parts.append(parts)
            except RuntimeError as exc:
                if is_oom(exc):
                    oom_events.append({"epoch": epoch, "error": str(exc)[:300]})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
        if not train_parts:
            raise RuntimeError("All strict-math training batches failed; likely OOM.")
        train_mean = {k: float(np.mean([p[k] for p in train_parts])) for k in train_parts[0]}
        dev_mean = evaluate_loss(model, dev_loader, device, **loss_kwargs)
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
    torch.save({"model_state": model.state_dict()}, out / "strict_math_lifting.pt")
    save_loss_curves(history, out)
    meta = {
        "created_at": now(),
        "model_name": model_name,
        "hidden_size": int(hidden_size),
        "observable_dim": int(observable_dim),
        "observable_multiplier": float(observable_dim / hidden_size),
        "uses_projection": False,
        "uses_pca_projection": False,
        "uses_random_projection": False,
        "uses_label_classifier": False,
        "uses_global_K_parameter": False,
        "dmd_rank": int(dmd_rank),
        "max_seq_len": int(max_seq_len),
        "batch_size": int(batch_size),
        "n_train_sequences_used": int(len(train_ds)),
        "n_dev_sequences_used": int(len(dev_ds)),
        "best_epoch": int(best[1]),
        "early_stopping_epoch": int(history[-1]["epoch"]) if history else None,
        "selection": "dev unsupervised reconstruction/local-DMD loss; no labels and no all_samples",
        "strict_math_failed_due_to_memory": bool(oom_events and not history),
        "oom_events": oom_events,
        "loss_weights": {
            "lambda_recon": lambda_recon,
            "lambda_dmd": lambda_dmd,
            "lambda_multistep": lambda_multistep,
            "lambda_stability": lambda_stability,
            "lambda_var": lambda_var,
        },
    }
    (out / "strict_math_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": model, "metadata": meta, "history": history}
