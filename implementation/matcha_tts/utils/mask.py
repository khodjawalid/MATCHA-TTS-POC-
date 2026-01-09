"""Masques de séquence : convention True = position valide, False = padding.

Les fonctions qui produisent des masques de padding explicitent le suffixe `_padding_mask`
avec True = padding. Pour l'entraînement/inférence, on privilégie les masques `text_mask`
et `mel_mask` où True = position valide et False = padding.
"""
from __future__ import annotations

import torch


def create_padding_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Construit un masque booléen (True = position masquée) à partir des longueurs effectives."""
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    mask = ids.unsqueeze(0) >= lengths.unsqueeze(1)
    return mask
