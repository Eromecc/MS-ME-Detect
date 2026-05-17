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
        for ax, key in zip(axes.ravel(), ["total", "recon", "lin", "multi"]):
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


def _as_zero(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros((), dtype=dtype, device=device)


def _loss_mode_weights(loss_mode: str, alpha: float, beta: float) -> tuple[float, float]:
    if loss_mode == "recon_only":
        return 0.0, 0.0
    if loss_mode == "recon_lin":
        return float(alpha), 0.0
    if loss_mode == "recon_lin_multi":
        return float(alpha), float(beta)
    raise ValueError(f"Unsupported strict Koopman loss_mode: {loss_mode}")


def _local_truncated_dmd_reduced(
    z: torch.Tensor,
    *,
    dmd_rank: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str | None]:
    """Differentiable per-document reduced DMD operator from current Z."""
    x = z[:-1].T.contiguous()
    y = z[1:].T.contiguous()
    if x.shape[1] < 2:
        raise ValueError("Need at least two transition snapshots for DMD.")
    try:
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        if s.numel() == 0:
            raise ValueError("empty singular spectrum")
        numerical_rank = int(torch.sum(s > (torch.max(s) * 1e-6)).detach().cpu().item())
        r = int(min(dmd_rank, numerical_rank, s.shape[0]))
        if r < 1:
            raise ValueError("rank too small")
        u_r = u[:, :r]
        s_r = s[:r]
        v_r = vh.conj().T[:, :r]
        k_tilde = u_r.T @ y @ v_r @ torch.diag(1.0 / (s_r + ridge))
        a = u_r.T @ x
        b = u_r.T @ y
        return k_tilde, u_r, a, b, None
    except RuntimeError as exc:
        # Keep a differentiable dynamics objective instead of silently dropping
        # the Koopman constraint when SVD is numerically unstable.
        r = int(min(dmd_rank, x.shape[0], x.shape[1]))
        if r < 1:
            raise
        a = x[:r, :]
        b = y[:r, :]
        gram = a @ a.T + float(ridge) * torch.eye(r, dtype=z.dtype, device=z.device)
        k_tilde = torch.linalg.solve(gram.T, (b @ a.T).T).T
        u_r = torch.eye(x.shape[0], r, dtype=z.dtype, device=z.device)
        return k_tilde, u_r, a, b, f"svd_fallback_ridge_lstsq:{type(exc).__name__}:{str(exc)[:160]}"


def compute_strict_koopman_losses(
    model: StrictKoopmanLifting,
    hidden_states: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    multi_steps: tuple[int, ...] = (2, 3),
    dmd_rank: int = 32,
    ridge: float = 1e-4,
    stability_weight: float = 0.0,
    return_details: bool = True,
    loss_mode: str = "recon_lin_multi",
    include_z: bool = False,
) -> tuple[torch.Tensor, dict[str, float | int | str | None]]:
    """Strict per-document Text-Koopman loss.

    K_tilde is estimated from the current document's lifted trajectory Z inside
    the forward loss path, so L_lin/L_multi gradients flow back to g_theta.
    """
    z, recon = model(hidden_states)
    recon_loss = F.mse_loss(recon, hidden_states)
    alpha_eff, beta_eff = _loss_mode_weights(loss_mode, alpha, beta)
    lin_loss = _as_zero(hidden_states.device, hidden_states.dtype)
    multi_loss = _as_zero(hidden_states.device, hidden_states.dtype)
    stability = _as_zero(hidden_states.device, hidden_states.dtype)
    details: dict[str, float | int | str | None] = {
        "recon": float(recon_loss.detach().cpu()),
        "lin": 0.0,
        "multi": 0.0,
        "stability": 0.0,
        "total": float(recon_loss.detach().cpu()),
        "valid_multi_steps": 0,
        "skipped_multi_steps": 0,
        "dmd_rank_used": 0,
        "dmd_warning": None,
    }
    if include_z:
        details["z"] = z
    if alpha_eff == 0.0 and beta_eff == 0.0 and float(stability_weight) == 0.0:
        return recon_loss, details if return_details else {}
    if z.shape[0] < 3:
        details["dmd_warning"] = f"sequence_too_short:{int(z.shape[0])}"
        return recon_loss, details if return_details else {}

    k_tilde, u_r, a, b, warning = _local_truncated_dmd_reduced(z, dmd_rank=dmd_rank, ridge=ridge)
    details["dmd_rank_used"] = int(k_tilde.shape[0])
    details["dmd_warning"] = warning
    b_pred = k_tilde @ a
    lin_loss = F.mse_loss(b_pred, b)

    valid_multi = []
    for step in tuple(int(s) for s in multi_steps):
        if step <= 1:
            details["skipped_multi_steps"] = int(details["skipped_multi_steps"]) + 1
            continue
        if z.shape[0] <= step + 1:
            details["skipped_multi_steps"] = int(details["skipped_multi_steps"]) + 1
            continue
        x_m = z[:-step].T.contiguous()
        y_m = z[step:].T.contiguous()
        # Use the same local DMD subspace/operator estimated from one-step
        # snapshots, matching the strict per-document Koopman formulation.
        if warning is None:
            a_m = u_r.T @ x_m
            b_m = u_r.T @ y_m
        else:
            r = k_tilde.shape[0]
            a_m = x_m[:r, :]
            b_m = y_m[:r, :]
        pred_m = torch.linalg.matrix_power(k_tilde, step) @ a_m
        valid_multi.append(F.mse_loss(pred_m, b_m))
    if valid_multi:
        multi_loss = torch.stack(valid_multi).mean()
    details["valid_multi_steps"] = len(valid_multi)
    if float(stability_weight) > 0.0:
        stability = _stability_loss(k_tilde)

    total = recon_loss + alpha_eff * lin_loss + beta_eff * multi_loss + float(stability_weight) * stability
    details.update(
        {
            "recon": float(recon_loss.detach().cpu()),
            "lin": float(lin_loss.detach().cpu()),
            "multi": float(multi_loss.detach().cpu()),
            "stability": float(stability.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
    )
    return total, details if return_details else {}


def batch_loss(
    model: StrictKoopmanLifting,
    batch,
    *,
    dmd_rank: int,
    alpha: float,
    beta: float,
    multi_steps: tuple[int, ...],
    ridge: float,
    stability_weight: float,
    lambda_var: float,
    loss_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=device)
    parts = {
        "recon": 0.0,
        "lin": 0.0,
        "multi": 0.0,
        "stability": 0.0,
        "var": 0.0,
        "valid_multi_steps": 0.0,
        "skipped_multi_steps": 0.0,
        "dmd_fallbacks": 0.0,
    }
    n = 0
    for _, h_cpu in batch:
        h = h_cpu.to(device, dtype=torch.float32)
        loss, detail = compute_strict_koopman_losses(
            model,
            h,
            alpha=alpha,
            beta=beta,
            multi_steps=multi_steps,
            dmd_rank=dmd_rank,
            ridge=ridge,
            stability_weight=stability_weight,
            loss_mode=loss_mode,
            return_details=True,
            include_z=True,
        )
        z = detail.pop("z")
        var = _variance_loss(z)
        var = _finite(var)
        loss = loss + lambda_var * var
        if not torch.isfinite(loss).all():
            continue
        total = total + loss
        parts["recon"] += float(detail["recon"])
        parts["lin"] += float(detail["lin"])
        parts["multi"] += float(detail["multi"])
        parts["stability"] += float(detail["stability"])
        parts["var"] += float(var.detach().cpu())
        parts["valid_multi_steps"] += float(detail["valid_multi_steps"])
        parts["skipped_multi_steps"] += float(detail["skipped_multi_steps"])
        parts["dmd_fallbacks"] += 1.0 if detail.get("dmd_warning") else 0.0
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
    resume: bool = False,
    alpha_lin: float = 1.0,
    beta_multi: float = 0.5,
    multi_steps: tuple[int, ...] = (2, 3),
    loss_mode: str = "recon_lin_multi",
    ridge: float = 1e-4,
    stability_weight: float = 0.0,
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
        "alpha": float(alpha_lin),
        "beta": float(beta_multi),
        "multi_steps": tuple(int(x) for x in multi_steps),
        "ridge": float(ridge),
        "stability_weight": float(stability_weight),
        "lambda_var": lambda_var,
        "loss_mode": loss_mode,
    }
    best = (np.inf, -1, None)
    bad = 0
    history: list[dict] = []
    oom_events: list[dict] = []
    start_epoch = 1
    last_ckpt = out / "strict_math_lifting_last.pt"
    if resume and last_ckpt.exists() and not (out / "strict_math_metadata.json").exists():
        payload = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            opt.load_state_dict(payload["optimizer_state"])
        history = list(payload.get("history", []))
        bad = int(payload.get("bad_epochs", 0))
        best_payload = payload.get("best")
        if best_payload:
            best = (float(best_payload["loss"]), int(best_payload["epoch"]), best_payload["model_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
    for epoch in range(start_epoch, int(epochs) + 1):
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
        print(
            f"[strict-math] epoch={epoch} loss_mode={loss_mode} dmd_rank={dmd_rank} "
            f"train_total={train_mean.get('total', np.nan):.6g} dev_total={dev_mean.get('total', np.nan):.6g}",
            flush=True,
        )
        if dev_mean["total"] < best[0]:
            best = (dev_mean["total"], epoch, {k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "history": history,
                "bad_epochs": bad,
                "best": {"loss": best[0], "epoch": best[1], "model_state": best[2]} if best[2] is not None else None,
            },
            last_ckpt,
        )
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
            "lambda_recon": 1.0,
            "alpha_lin": alpha_lin,
            "beta_multi": beta_multi,
            "stability_weight": stability_weight,
            "lambda_var": lambda_var,
        },
        "loss_mode": loss_mode,
        "multi_steps": list(multi_steps),
        "ridge": float(ridge),
    }
    (out / "strict_math_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": model, "metadata": meta, "history": history}
