"""Transition-state profiling features from token-level loss sequences.

The module intentionally avoids raw token-id transitions. It maps token losses
or token probabilities to abstract states, then summarizes transition patterns.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
    from .feature_probability import load_causal_lm
    from .utils import write_csv
except ImportError:
    import config
    from feature_probability import load_causal_lm
    from utils import write_csv

EPS = 1e-12


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return np.nan
    p = counts.astype(float) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def encode_quantile_states(values: np.ndarray, n_states: int = 5) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=int)
    if np.nanmax(vals) == np.nanmin(vals):
        return np.zeros(vals.size, dtype=int)
    qs = np.nanquantile(vals, np.linspace(0, 1, n_states + 1)[1:-1])
    return np.searchsorted(qs, vals, side="right").astype(int)


def fit_loss_bins(sequences: list[np.ndarray], n_states_list: list[int] | None = None) -> dict[str, list[float]]:
    n_states_list = n_states_list or [3, 5, 7]
    vals = np.concatenate([np.asarray(s, dtype=float)[np.isfinite(s)] for s in sequences if len(s)]) if sequences else np.asarray([])
    bins: dict[str, list[float]] = {}
    if vals.size == 0:
        return {str(n): [] for n in n_states_list}
    for n_states in n_states_list:
        if np.nanmax(vals) == np.nanmin(vals):
            bins[str(n_states)] = []
        else:
            bins[str(n_states)] = [float(x) for x in np.nanquantile(vals, np.linspace(0, 1, n_states + 1)[1:-1])]
    return bins


def encode_with_bins(values: np.ndarray, bins: list[float], n_states: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=int)
    if not bins:
        return np.zeros(vals.size, dtype=int)
    return np.clip(np.searchsorted(np.asarray(bins, dtype=float), vals, side="right"), 0, n_states - 1).astype(int)


def encode_probability_rank_states(losses: np.ndarray, n_states: int = 5) -> np.ndarray:
    # Higher probability corresponds to lower loss. State 0 = lowest probability/highest loss.
    vals = -np.asarray(losses, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([], dtype=int)
    ranks = pd.Series(vals).rank(method="average", pct=True).to_numpy()
    states = np.floor(ranks * n_states).astype(int)
    return np.clip(states, 0, n_states - 1)


def run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    cur = 0
    for item in mask:
        if bool(item):
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


def transition_features_from_states(states: np.ndarray, prefix: str, n_states: int = 5) -> dict[str, float]:
    states = np.asarray(states, dtype=int)
    out: dict[str, float] = {}
    if states.size == 0:
        for i in range(n_states):
            for j in range(n_states):
                out[f"{prefix}_trans_{i}_to_{j}"] = np.nan
        for name in [
            "transition_entropy",
            "self_transition_rate",
            "upward_transition_rate",
            "downward_transition_rate",
            "large_jump_rate",
            "high_loss_burst_density",
            "low_loss_run_length_mean",
            "high_loss_run_length_mean",
            "state_sequence_entropy",
            "transition_spectral_gap",
            "token_count",
        ]:
            out[f"{prefix}_{name}"] = np.nan
        return out

    trans = np.zeros((n_states, n_states), dtype=float)
    if states.size >= 2:
        for a, b in zip(states[:-1], states[1:]):
            if 0 <= a < n_states and 0 <= b < n_states:
                trans[a, b] += 1
    trans_total = trans.sum()
    trans_norm = trans / trans_total if trans_total > 0 else np.full_like(trans, np.nan)
    for i in range(n_states):
        for j in range(n_states):
            out[f"{prefix}_trans_{i}_to_{j}"] = float(trans_norm[i, j]) if trans_total > 0 else np.nan
    deltas = np.diff(states) if states.size >= 2 else np.asarray([], dtype=int)
    high = states >= n_states - 1
    low = states <= 0
    out[f"{prefix}_transition_entropy"] = entropy_from_counts(trans.ravel())
    out[f"{prefix}_self_transition_rate"] = float(np.mean(deltas == 0)) if deltas.size else np.nan
    out[f"{prefix}_upward_transition_rate"] = float(np.mean(deltas > 0)) if deltas.size else np.nan
    out[f"{prefix}_downward_transition_rate"] = float(np.mean(deltas < 0)) if deltas.size else np.nan
    out[f"{prefix}_large_jump_rate"] = float(np.mean(np.abs(deltas) >= 2)) if deltas.size else np.nan
    out[f"{prefix}_high_loss_burst_density"] = float(np.mean(high)) if states.size else np.nan
    low_runs = run_lengths(low)
    high_runs = run_lengths(high)
    out[f"{prefix}_low_loss_run_length_mean"] = float(np.mean(low_runs)) if low_runs else 0.0
    out[f"{prefix}_high_loss_run_length_mean"] = float(np.mean(high_runs)) if high_runs else 0.0
    out[f"{prefix}_state_sequence_entropy"] = entropy_from_counts(np.bincount(states, minlength=n_states))
    try:
        row_sums = trans.sum(axis=1, keepdims=True)
        pmat = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums > 0)
        eig = np.sort(np.abs(np.linalg.eigvals(pmat)))[::-1]
        out[f"{prefix}_transition_spectral_gap"] = float(1.0 - eig[1]) if len(eig) > 1 else np.nan
    except Exception:
        out[f"{prefix}_transition_spectral_gap"] = np.nan
    out[f"{prefix}_token_count"] = float(states.size)
    return out


def transition_features_from_losses(losses: list[float] | np.ndarray, prefix: str, n_states: int = 5) -> dict[str, float]:
    vals = np.asarray(losses, dtype=float)
    vals = vals[np.isfinite(vals)]
    out = {}
    out.update(transition_features_from_states(encode_quantile_states(vals, n_states), f"{prefix}_lossq", n_states))
    out.update(transition_features_from_states(encode_probability_rank_states(vals, n_states), f"{prefix}_probrank", n_states))
    return out


def transition_features_from_losses_with_bins(losses: list[float] | np.ndarray, prefix: str, bins_by_state: dict[str, list[float]]) -> dict[str, float]:
    vals = np.asarray(losses, dtype=float)
    vals = vals[np.isfinite(vals)]
    out = {}
    for key, bins in bins_by_state.items():
        n_states = int(key)
        out.update(transition_features_from_states(encode_with_bins(vals, bins, n_states), f"{prefix}_lossq{n_states}", n_states))
    # Probability-rank states are per-sample ranks and do not use external labels.
    for n_states in [3, 5, 7]:
        out.update(transition_features_from_states(encode_probability_rank_states(vals, n_states), f"{prefix}_probrank{n_states}", n_states))
    return out


def token_loss_sequence(text: str, tokenizer, model, max_length: int = 512) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not str(text).strip():
        return np.asarray([], dtype=float)
    device = next(model.parameters()).device
    encoded = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if input_ids.shape[1] < 2:
        return np.asarray([], dtype=float)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    return losses.detach().float().cpu().numpy()


def parse_sequence_cell(value) -> list[float] | None:
    if pd.isna(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    try:
        return [float(x) for x in text.split() if x]
    except Exception:
        return None


def discover_token_loss_cache(paths: list[str | Path]) -> tuple[Path | None, list[str]]:
    sequence_markers = ["loss_sequence", "token_losses", "token_loss_sequence", "losses_json"]
    for raw in paths:
        path = Path(raw)
        candidates = [path] if path.is_file() else list(path.rglob("*.csv")) + list(path.rglob("*.jsonl")) + list(path.rglob("*.jsonl.gz")) if path.exists() else []
        for candidate in candidates:
            try:
                if candidate.suffix == ".csv":
                    cols = pd.read_csv(candidate, nrows=0).columns.tolist()
                elif candidate.name.endswith(".jsonl.gz"):
                    with gzip.open(candidate, "rt", encoding="utf-8") as handle:
                        first = handle.readline()
                    cols = list(json.loads(first).keys())
                elif candidate.suffix == ".jsonl":
                    first = candidate.read_text(encoding="utf-8").splitlines()[0]
                    cols = list(json.loads(first).keys())
                else:
                    continue
            except Exception:
                continue
            seq_cols = [c for c in cols if any(marker in c.lower() for marker in sequence_markers)]
            if seq_cols and "id" in cols:
                return candidate, seq_cols
    return None, []


def build_from_cache(cache_path: Path, seq_cols: list[str], output_path: str | Path, n_states: int = 5) -> pd.DataFrame:
    if cache_path.suffix == ".csv":
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_json(cache_path, lines=True)
    rows = []
    for _, row in df.iterrows():
        feats = {"id": row["id"]}
        for col in seq_cols:
            seq = parse_sequence_cell(row[col])
            if seq is None:
                continue
            prefix = col.replace("_loss_sequence", "").replace("_token_losses", "").replace("_losses_json", "")
            feats.update(transition_features_from_losses(seq, prefix=prefix, n_states=n_states))
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def iter_token_loss_cache(cache_path: str | Path):
    path = Path(cache_path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_loss_sequences(cache_path: str | Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for item in iter_token_loss_cache(cache_path):
        out[str(item["id"])] = np.asarray(item.get("loss_sequence", []), dtype=float)
    return out


def build_transition_features_from_loss_cache(
    cache_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    bins_by_state: dict[str, list[float]],
) -> pd.DataFrame:
    rows = []
    for item in iter_token_loss_cache(cache_path):
        losses = np.asarray(item.get("loss_sequence", []), dtype=float)
        feats = {"id": item["id"]}
        feats.update(transition_features_from_losses_with_bins(losses, prefix=model_name, bins_by_state=bins_by_state))
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def build_smoke_features(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_key: str = "small",
    sample_size: int = 80,
    seed: int = 42,
    max_length: int = 384,
    n_states: int = 5,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if sample_size and len(df) > sample_size:
        stratify_cols = ["label"] if "label" in df.columns else None
        df = df.groupby(stratify_cols, group_keys=False).apply(lambda g: g.sample(n=min(len(g), max(1, sample_size // max(df["label"].nunique(), 1))), random_state=seed)) if stratify_cols else df.sample(n=sample_size, random_state=seed)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=seed)
    local_path = config.get_model_local_path(model_key)
    if not config.is_local_model_ready(local_path):
        raise FileNotFoundError(f"Local model for {model_key} is not ready at {local_path}; not downloading.")
    tokenizer, model = load_causal_lm(local_path, dtype=config.DTYPE, device_map=None, local_files_only=True)
    prefix = config.MODEL_KEY_PREFIX.get(model_key, model_key)
    rows = []
    for _, row in df.iterrows():
        losses = token_loss_sequence(row["text"], tokenizer, model, max_length=max_length)
        feats = {"id": row["id"]}
        feats.update(transition_features_from_losses(losses, prefix=prefix, n_states=n_states))
        rows.append(feats)
    out = pd.DataFrame(rows)
    write_csv(out, output_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache_paths", nargs="*", default=[])
    parser.add_argument("--model_key", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--sample_size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=384)
    args = parser.parse_args()
    cache, seq_cols = discover_token_loss_cache(args.cache_paths)
    if cache is not None:
        print(f"Using token-level loss cache: {cache}")
        build_from_cache(cache, seq_cols, args.output)
    else:
        print("No token-level loss cache found; running small local-model smoke feature extraction.")
        build_smoke_features(args.input, args.output, model_key=args.model_key, sample_size=args.sample_size, seed=args.seed, max_length=args.max_length)


if __name__ == "__main__":
    main()
