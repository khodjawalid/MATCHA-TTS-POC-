"""Encodeur de texte avec attention RoPE pour contextualiser les séquences de phonèmes."""
from __future__ import annotations

import torch
import torch.nn as nn

from .rope import apply_rotary_pos_emb, build_rotary_frequencies


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Attention multi-tête qui applique RoPE sur q/k avant le produit scalaire."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert dim % num_heads == 0, "head_dim must divide dim"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Calcule l'attention self avec RoPE et masque de padding.

        Args:
            x: (B, T, D)
            padding_mask: (B, T) bool, True pour positions à masquer.
        """

        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)

        rotary = build_rotary_frequencies(T, self.head_dim, base=10000).to(x.device)
        q, k = apply_rotary_pos_emb(q, k, rotary)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if padding_mask is not None:
            # padding_mask: True pour les pads → -inf dans les scores
            mask = padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    """Bloc Transformer : attention RoPE + feed-forward + résidus + dropout."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttentionRoPE(dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Applique attention RoPE puis feed-forward avec connexions résiduelles."""
        attn_out = self.attn(self.norm1(x), padding_mask)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


class TextEncoder(nn.Module):
    """Produit des représentations contextuelles à partir de séquences de phonèmes (IDs)."""

    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        """Initialise embeddings et pile de blocs Transformer avec RoPE."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(embedding_dim)

    def forward(self, phoneme_ids: torch.Tensor, phoneme_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode une séquence de phonèmes en représentations contextuelles.

        Args:
            phoneme_ids: (B, T_text) entiers indexant le vocabulaire de phonèmes.
            phoneme_mask: (B, T_text) bool, True = valide.
        Returns:
            encoded_text: (B, T_text, D) représentations contextualisées.
        """

        x = self.embedding(phoneme_ids)
        x = self.dropout(x)
        # phoneme_mask: True = valide, convertir pour key_padding_mask (True = padding)
        padding_mask = None
        if phoneme_mask is not None:
            padding_mask = ~phoneme_mask
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.norm_out(x)


if __name__ == "__main__":
    # Test rapide : vérifie les shapes pour un batch aléatoire
    batch = 2
    seq = 5
    vocab = 32
    dim = 16
    x = torch.randint(0, vocab, (batch, seq))
    mask = x != 0  # 0 = padding → mask True sur positions valides

    encoder = TextEncoder(vocab_size=vocab, embedding_dim=dim, num_layers=2, num_heads=4, dropout=0.1)
    out = encoder(x, mask)
    print("Input:", x.shape, "Output:", out.shape)
