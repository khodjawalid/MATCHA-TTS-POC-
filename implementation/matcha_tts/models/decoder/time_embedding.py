"""Embeddings temporels sinusoidaux + projection MLP pour le flow."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Génère un embedding sinusoïdal type diffusion.

    On encode t dans [0,1] (ou [0,T]) en paires sin/cos, ce qui fournit au modèle une
    notion continue de la progression de la trajectoire de flow. Pas de paramètres appris
    ici : la phase est fixe et stable, ce qui suffit pour le conditionnement temporel.
    """

    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / (half_dim - 1)))
    angles = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, emb[:, :1]], dim=-1)
    return emb


def build_time_mlp(dim: int) -> nn.Module:
    """Petit MLP pour projeter l'embedding temporel avant injection dans les blocs ResNet."""
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.SiLU(),
        nn.Linear(dim, dim),
    )
