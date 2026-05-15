"""Datasets and observables for Deep DMD / Koopman encoder experiments.

Theory mapping:
- input token trajectory: x_t = content-abstracted token-loss observable
- learnable lifting: z_t = g_theta(x_t), implemented in deep_dmd_model.py
- no raw token IDs or token strings are used here
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def rolling_mean(vals: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(vals).rolling(window, min_periods=1, center=True).mean().to_numpy(dtype=float)


def rolling_std(vals: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(vals).rolling(window, min_periods=2, center=True).std().fillna(0.0).to_numpy(dtype=float)


def read_token_loss_cache(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            out[str(item["id"])] = item
    return out


def fit_loss_bins(cache_items: dict[str, dict], ids: list[str], n_states: int = 5) -> list[float]:
    vals = []
    for rid in ids:
        item = cache_items.get(str(rid))
        if not item:
            continue
        seq = np.asarray(item.get("loss_sequence", []), dtype=float)
        seq = seq[np.isfinite(seq)]
        if seq.size:
            vals.append(seq)
    if not vals:
        return []
    all_vals = np.concatenate(vals)
    if np.nanmax(all_vals) == np.nanmin(all_vals):
        return []
    return [float(x) for x in np.nanquantile(all_vals, np.linspace(0, 1, n_states + 1)[1:-1])]


def make_observable(item: dict, *, max_seq_len: int, loss_bins: list[float]) -> np.ndarray:
    loss = np.asarray(item.get("loss_sequence", []), dtype=float)
    loss = loss[np.isfinite(loss)]
    if loss.size == 0:
        return np.zeros((0, 10), dtype=np.float32)
    if max_seq_len and loss.size > max_seq_len:
        loss = loss[:max_seq_len]
    delta = np.diff(loss, prepend=loss[0])
    sd = float(np.std(loss))
    zloss = (loss - float(np.mean(loss))) / (sd if sd > 0 else 1.0)
    if loss_bins:
        state = np.clip(np.searchsorted(np.asarray(loss_bins), loss, side="right"), 0, len(loss_bins)).astype(float)
        state = state / max(len(loss_bins), 1)
    else:
        state = np.zeros_like(loss, dtype=float)
    pos = np.linspace(0.0, 1.0, loss.size, dtype=float)
    cols = [
        loss,
        zloss,
        delta,
        np.abs(delta),
        rolling_mean(loss, 5),
        rolling_std(loss, 5),
        rolling_mean(loss, 11),
        rolling_std(loss, 11),
        state,
        pos,
    ]
    rank = item.get("rank_sequence")
    prob = item.get("prob_sequence")
    if rank is not None and prob is not None:
        r = np.asarray(rank, dtype=float)[: loss.size]
        p = np.asarray(prob, dtype=float)[: loss.size]
        if r.size == loss.size and p.size == loss.size:
            rlog = np.log1p(np.maximum(r, 0.0))
            cols.extend([rlog, p, np.diff(rlog, prepend=rlog[0])])
    return np.vstack(cols).T.astype(np.float32)


@dataclass
class DeepDMDScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "DeepDMDScaler":
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32))


def fit_observable_scaler(cache_items: dict[str, dict], ids: list[str], *, max_seq_len: int, loss_bins: list[float]) -> DeepDMDScaler:
    chunks = []
    for rid in ids:
        item = cache_items.get(str(rid))
        if not item:
            continue
        x = make_observable(item, max_seq_len=max_seq_len, loss_bins=loss_bins)
        if len(x):
            chunks.append(x)
    if not chunks:
        return DeepDMDScaler(np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32))
    all_x = np.concatenate(chunks, axis=0)
    mean = np.nanmean(all_x, axis=0).astype(np.float32)
    std = np.nanstd(all_x, axis=0).astype(np.float32)
    std[std <= 1e-6] = 1.0
    return DeepDMDScaler(mean, std)


class DeepDMDTokenDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        cache_items: dict[str, dict],
        *,
        max_seq_len: int,
        loss_bins: list[float],
        scaler: DeepDMDScaler,
        min_tokens: int = 20,
    ) -> None:
        self.rows = []
        self.skipped_ids = []
        self.input_dim = len(scaler.mean)
        for _, row in meta.iterrows():
            rid = str(row["id"])
            item = cache_items.get(rid)
            if not item:
                self.skipped_ids.append(rid)
                continue
            x = make_observable(item, max_seq_len=max_seq_len, loss_bins=loss_bins)
            if len(x) < min_tokens:
                self.skipped_ids.append(rid)
                continue
            x = scaler.transform(x)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            self.rows.append(
                {
                    "id": rid,
                    "x": x,
                    "label": int(row["label"]),
                    "source_dataset": row.get("source_dataset", ""),
                    "domain": row.get("domain", ""),
                    "generator": row.get("generator", ""),
                    "transition_split": row.get("transition_split", ""),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_deep_dmd(batch: list[dict]) -> dict:
    max_len = max(item["x"].shape[0] for item in batch)
    dim = batch[0]["x"].shape[1]
    x = np.zeros((len(batch), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=bool)
    labels = np.zeros(len(batch), dtype=np.float32)
    ids = []
    meta = []
    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x[i, :n] = item["x"]
        mask[i, :n] = True
        labels[i] = item["label"]
        ids.append(item["id"])
        meta.append({k: item.get(k, "") for k in ["source_dataset", "domain", "generator", "transition_split"]})
    return {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "label": torch.from_numpy(labels),
        "id": ids,
        "meta": meta,
    }
