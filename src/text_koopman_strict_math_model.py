"""Strict mathematical Text-Koopman lifting model.

This variant intentionally avoids PCA, random projection, pooled latent
classification, and global Koopman parameters.  Hidden-state token trajectories
enter the lifting map directly:

    H[t] in R^hidden_size -> g_theta(H[t]) in R^observable_dim

Downstream classifiers must use only per-document local-DMD spectral summaries.
"""

from __future__ import annotations

import torch
from torch import nn


class StrictKoopmanLifting(nn.Module):
    """Learned lifting g_theta: hidden_size -> observable_dim."""

    def __init__(self, hidden_size: int, observable_dim: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.observable_dim = int(observable_dim)
        if self.observable_dim <= self.hidden_size:
            raise ValueError("Strict mathematical Text-Koopman requires observable_dim > hidden_size.")
        self.g = nn.Sequential(
            nn.Linear(self.hidden_size, self.observable_dim),
            nn.GELU(),
            nn.LayerNorm(self.observable_dim),
        )
        self.decoder = nn.Sequential(nn.Linear(self.observable_dim, self.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.g(hidden_states)
        recon = self.decoder(z)
        return z, recon
