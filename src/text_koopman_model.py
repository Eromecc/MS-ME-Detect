"""Strict Text-Koopman lifting model.

This module contains no label classifier. The learned lifting network maps
projected hidden-state trajectories to an observable space. Downstream
classification must use only local-K spectral/residual scalar features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HiddenProjector:
    """Train-only scaler plus dimensionality projection.

    The stored mean/std are estimated from public train tokens only. The
    projection can be random Gaussian, train-only PCA, or a trainable linear
    layer inside ``TextKoopmanLifting``.
    """

    def __init__(self, hidden_size: int, projection_dim: int, seed: int = 42, projector_type: str = "random") -> None:
        self.hidden_size = int(hidden_size)
        self.projection_dim = int(projection_dim)
        self.projector_type = str(projector_type)
        if self.projector_type not in {"random", "pca", "linear"}:
            raise ValueError("projector_type must be 'random', 'pca', or 'linear'.")
        rng = np.random.default_rng(seed)
        mat = rng.normal(0.0, 1.0 / np.sqrt(self.projection_dim), size=(self.hidden_size, self.projection_dim))
        self.projection = mat.astype("float32")
        self.mean = np.zeros(self.hidden_size, dtype="float32")
        self.std = np.ones(self.hidden_size, dtype="float32")
        self.explained_variance_ratio = None

    def fit_scaler(self, hidden_arrays: list[np.ndarray], max_tokens: int = 200_000) -> "HiddenProjector":
        chunks = []
        n = 0
        for arr in hidden_arrays:
            if arr.size == 0:
                continue
            take = arr
            remaining = max_tokens - n
            if remaining <= 0:
                break
            if len(take) > remaining:
                idx = np.linspace(0, len(take) - 1, remaining).astype(int)
                take = take[idx]
            chunks.append(take.astype("float32"))
            n += len(take)
        if chunks:
            x = np.concatenate(chunks, axis=0)
            self.mean = x.mean(axis=0).astype("float32")
            self.std = x.std(axis=0).astype("float32")
            self.std[self.std < 1e-6] = 1.0
            if self.projector_type == "pca":
                from sklearn.decomposition import PCA

                scaled = (x.astype("float32") - self.mean) / self.std
                pca = PCA(n_components=self.projection_dim, svd_solver="randomized", random_state=0)
                pca.fit(scaled)
                self.projection = pca.components_.T.astype("float32")
                self.explained_variance_ratio = pca.explained_variance_ratio_.astype("float32")
        return self

    def transform_numpy(self, hidden: np.ndarray) -> np.ndarray:
        x = (hidden.astype("float32") - self.mean) / self.std
        if self.projector_type == "linear":
            return x.astype("float32")
        return (x @ self.projection).astype("float32")

    def transform_tensor(self, hidden: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        target = device or hidden.device
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=target)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=target)
        x = (hidden.to(target, dtype=torch.float32) - mean) / std
        if self.projector_type == "linear":
            return x
        proj = torch.as_tensor(self.projection, dtype=torch.float32, device=target)
        return x @ proj


class TextKoopmanLifting(nn.Module):
    """Learnable lifting g_theta: projected hidden state -> observable state."""

    def __init__(
        self,
        projection_dim: int,
        observable_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if observable_dim <= projection_dim:
            raise ValueError("Strict Text-Koopman requires observable_dim > projection_dim.")
        self.projection_dim = int(projection_dim)
        self.input_dim = int(input_dim or projection_dim)
        self.observable_dim = int(observable_dim)
        self.input_projector = nn.Identity() if self.input_dim == self.projection_dim else nn.Linear(self.input_dim, self.projection_dim)
        self.g = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, observable_dim),
            nn.GELU(),
            nn.LayerNorm(observable_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(observable_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def forward(self, projected_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projector(projected_hidden)
        z = self.g(projected)
        recon = self.decoder(z)
        return z, recon
