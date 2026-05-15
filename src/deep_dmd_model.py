"""Learnable lifting and Koopman operator for Deep DMD experiments."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        act = nn.GELU if activation == "gelu" else nn.ReLU
        layers = []
        dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(dim, hidden_dim), act(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepDMDEncoder(nn.Module):
    """Time-distributed MLP lifting g_theta and learnable linear K.

    Input trajectory x_t is supplied by deep_dmd_dataset.py. The model learns
    z_t = g_theta(x_t), then enforces z_{t+1} ~= K z_t with multi-step losses.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, latent_dim, hidden_dim, num_layers, dropout, activation)
        self.K = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))
        self.decoder = MLP(latent_dim, input_dim, hidden_dim, 1, dropout, activation) if use_decoder else None
        pooled_dim = latent_dim * 4
        self.classifier = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        z = self.encoder(x.reshape(b * t, d)).reshape(b, t, self.latent_dim)
        return z

    def pooled(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        mean = (z * m).sum(dim=1) / denom
        var = (((z - mean.unsqueeze(1)) ** 2) * m).sum(dim=1) / denom
        std = torch.sqrt(var.clamp_min(1e-8))
        lengths = mask.sum(dim=1).clamp_min(1)
        last = z[torch.arange(z.shape[0], device=z.device), lengths - 1]
        first = z[:, 0]
        return torch.cat([mean, std, first, last], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        z = self.lift(x)
        logits = self.classifier(self.pooled(z, mask)).squeeze(-1)
        recon = None
        if self.decoder is not None:
            b, t, d = z.shape
            recon = self.decoder(z.reshape(b * t, d)).reshape(b, t, self.input_dim)
        return {"z": z, "logits": logits, "recon": recon, "K": self.K}

    def apply_k(self, z: torch.Tensor, steps: int = 1) -> torch.Tensor:
        k = torch.matrix_power(self.K, steps)
        return z @ k.T
