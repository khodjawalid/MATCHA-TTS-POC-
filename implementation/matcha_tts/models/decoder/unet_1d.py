"""UNet 1D conditionné pour prédire le champ de vitesse v_theta."""
from __future__ import annotations

import torch
import torch.nn as nn

from .resnet_block import ResNetBlock
from .transformer_block import TransformerBlock
from .time_embedding import sinusoidal_time_embedding, build_time_mlp


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Ajuste un tenseur en recadrant ou en paddant à droite pour atteindre target_len."""

    current_len = x.shape[-1]
    if current_len == target_len:
        return x

    diff = target_len - current_len
    if diff > 0:
        pad_shape = list(x.shape[:-1]) + [diff]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    return x[..., :target_len]


class UNet1D(nn.Module):
    """Architecture UNet 1D avec skips, attention et injection temporelle."""

    def __init__(
        self,
        channels: int,
        attention_heads: int,
        time_embedding_dim: int,
        hidden_dim: int,
        num_layers_per_block: int,
        num_down_blocks: int,
        num_mid_blocks: int,
        num_up_blocks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_proj = nn.Conv1d(channels * 2, hidden_dim, kernel_size=1)

        self.time_dim = time_embedding_dim
        self.time_mlp = build_time_mlp(time_embedding_dim)

        def make_res_block():
            return ResNetBlock(hidden_dim, time_dim=time_embedding_dim, dropout=dropout)

        def make_transformer():
            return TransformerBlock(hidden_dim, num_heads=attention_heads, dropout=dropout)

        self.down = nn.ModuleList([
            nn.ModuleList([make_res_block() for _ in range(num_layers_per_block)] + [make_transformer()])
            for _ in range(num_down_blocks)
        ])

        self.mid = nn.ModuleList([
            nn.ModuleList([make_res_block() for _ in range(num_layers_per_block)] + [make_transformer()])
            for _ in range(num_mid_blocks)
        ])

        self.up = nn.ModuleList([
            nn.ModuleList([make_res_block() for _ in range(num_layers_per_block)] + [make_transformer()])
            for _ in range(num_up_blocks)
        ])
        self.skip_projs = nn.ModuleList([
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1) for _ in range(num_up_blocks)
        ])

        self.out_norm = nn.GroupNorm(8, hidden_dim)
        self.out_act = nn.SiLU()
        self.out = nn.Conv1d(hidden_dim, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prédit v_theta(x, t | µ).

        Args:
            x: (B, C, T) état courant de la trajectoire.
            time: (B,) pas temporel [0,1].
            conditioning: (B, C, T) µ aligné temporellement.
            mel_mask: (B, T) bool, True = valide, False = padding.
        Returns:
            v: (B, C, T) champ de vitesse.
        """

        if conditioning is None:
            raise ValueError("conditioning (mu) est requis pour le décodeur")

        if mel_mask is not None and mel_mask.shape[-1] != x.shape[-1]:
            raise ValueError("mel_mask doit correspondre à la longueur temporelle de x")

        original_len = x.shape[-1]

        # Concatène x et µ sur les canaux, projette
        h = torch.cat([x, conditioning], dim=1)
        h = self.in_proj(h)

        # Embedding temporel
        t_sin = sinusoidal_time_embedding(time, self.time_dim)
        t_emb = self.time_mlp(t_sin)  # (B, time_dim)

        current_mask = mel_mask

        skips = []
        for block_group in self.down:
            for blk in block_group[:-1]:
                h = blk(h, t_emb)
            h = block_group[-1](h, mask=current_mask)
            skips.append(h)
            # ceil_mode décale les longueurs pour les T impairs; on reste sur floor pour aligner les skips
            h = torch.nn.functional.avg_pool1d(h, kernel_size=2, stride=2, ceil_mode=False)

            if current_mask is not None:
                # max_pool conserve la validité si au moins un élément de la fenêtre est vrai
                pooled_mask = torch.nn.functional.max_pool1d(
                    current_mask.float().unsqueeze(1), kernel_size=2, stride=2, ceil_mode=False
                )
                current_mask = (pooled_mask > 0.0).squeeze(1)

        for block_group in self.mid:
            for blk in block_group[:-1]:
                h = blk(h, t_emb)
            h = block_group[-1](h, mask=current_mask)

        for idx, block_group in enumerate(self.up):
            h = torch.nn.functional.interpolate(h, scale_factor=2, mode="nearest")
            if current_mask is not None:
                up_mask = torch.nn.functional.interpolate(
                    current_mask.float().unsqueeze(1), scale_factor=2, mode="nearest"
                )
                current_mask = (up_mask > 0.5).squeeze(1)
            if skips:
                skip = skips.pop()
                h = _match_length(h, skip.shape[-1])
                if current_mask is not None:
                    current_mask = _match_length(current_mask, skip.shape[-1])
                h = torch.cat([h, skip], dim=1)
                h = self.skip_projs[idx](h)
            for blk in block_group[:-1]:
                h = blk(h, t_emb)
            h = block_group[-1](h, mask=current_mask)

        h = _match_length(h, original_len)
        h = self.out(h)
        return h
