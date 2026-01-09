"""Bloc résiduel 1D avec injection de time embedding pour le flow."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Bloc résiduel convolutionnel avec normalisation et temps injecté."""

    def __init__(self, channels: int, time_dim: int, dropout: float = 0.0) -> None:
        """Deux conv1d + normalisation + activation, et ajout du time embedding projeté."""
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, channels)

        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Injecte t_emb après la première conv, retourne la sortie résiduelle."""
        h = self.conv1(self.act1(self.norm1(x)))
        t_added = self.time_proj(t_emb).unsqueeze(-1)  # (B, C, 1)
        h = h + t_added
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return x + h
