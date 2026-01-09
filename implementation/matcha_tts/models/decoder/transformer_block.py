"""Bloc Transformer 1D sans embeddings positionnels, activation snakebeta."""
from __future__ import annotations

import torch
import torch.nn as nn


class SnakeBeta(nn.Module):
    """Activation périodique douce utilisée dans BigVGAN pour enrichir le spectre fréquentiel."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(self.beta * x) ** 2 / self.beta


class TransformerBlock(nn.Module):
    """Self-attention sans positions, adaptée au conditionnement µ (déjà aligné temporellement)."""

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            SnakeBeta(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Applique self-attention masquée puis feed-forward snakebeta."""
        # mask: True = valide, False = padding (converti pour MultiheadAttention)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        # x: (B, C, T) -> (B, T, C) pour MultiheadAttention
        x_t = x.transpose(1, 2)
        attn_out, _ = self.attn(x_t, x_t, x_t, key_padding_mask=key_padding_mask)
        x_t = x_t + self.dropout(attn_out)
        x_t = self.norm1(x_t)
        ff_out = self.ff(x_t)
        x_t = self.norm2(x_t + ff_out)
        return x_t.transpose(1, 2)
