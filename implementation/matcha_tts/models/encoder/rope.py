"""Implémentation du Rotary Position Embedding (RoPE) pour encoder l'information de position."""
from __future__ import annotations

import math
import torch


def build_rotary_frequencies(seq_len: int, dim: int, base: int = 10000) -> torch.Tensor:
    """Construit les fréquences sinusoïdales utilisées par RoPE.

    RoPE encode la position en appliquant une rotation dépendante du rang aux sous-dimensions
    paires/impaire des tenseurs (q, k). Contrairement aux positional embeddings absolus,
    RoPE n'ajoute pas un vecteur de position : la rotation préserve la norme et rend
    l'attention sensible aux décalages relatifs de manière plus lisse.
    """

    theta = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)[:, None]
    freqs = positions * theta[None, :]
    sin = freqs.sin()
    cos = freqs.cos()
    # Retourne shape (seq_len, dim/2, 2) implicite via sin/cos séparés
    return torch.stack((sin, cos), dim=-1)  # (seq_len, dim/2, 2)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, rotary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applique la rotation RoPE sur q et k avant le produit scalaire d'attention.

    Args:
        q: (B, num_heads, T, head_dim)
        k: (B, num_heads, T, head_dim)
        rotary: sin/cos pré-calculés de shape (T, head_dim/2, 2)

    Returns:
        q_rot, k_rot avec positions incorporées.
    """

    # Sépare les moitiés pair/impair pour appliquer la rotation complexe (x_even, x_odd)
    sin = rotary[:, :, 0]  # (T, head_dim/2)
    cos = rotary[:, :, 1]  # (T, head_dim/2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    q_rot = rotate(q)
    k_rot = rotate(k)
    return q_rot, k_rot
