"""
Tests de robustesse longueur du UNet 1D : sortie strictement alignée à l'entrée
pour des longueurs paires et impaires, avec masque de padding.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.decoder.unet_1d import UNet1D


LENGTHS = [63, 64, 65, 127, 128, 129]
BATCH = 2
N_MELS = 80


def _build_model() -> UNet1D:
    return UNet1D(
        channels=N_MELS,
        attention_heads=2,
        time_embedding_dim=256,
        hidden_dim=256,
        num_layers_per_block=1,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout=0.1,
    )


def _make_inputs(T: int):
    x = torch.randn(BATCH, N_MELS, T, requires_grad=True)
    mu = torch.randn(BATCH, N_MELS, T)
    mask = torch.ones(BATCH, T, dtype=torch.bool)
    pad = min(5, max(1, T // 8))
    # Padding en fin pour vérifier l'alignement masque/signal
    mask[:, -pad:] = False
    t = torch.rand(BATCH)
    return x, mu, mask, t


def test_unet_preserves_length_and_backward():
    torch.manual_seed(0)
    model = _build_model()
    model.train()

    for T in LENGTHS:
        x, mu, mask, t = _make_inputs(T)
        model.zero_grad(set_to_none=True)

        v = model(x, t, conditioning=mu, mel_mask=mask)

        assert v.shape == x.shape, f"Longueur modifiée pour T={T}"
        assert torch.isfinite(v).all(), f"NaN/Inf dans la sortie pour T={T}"

        loss = (v ** 2).mean()
        loss.backward()

        grad_norm = x.grad.norm().item()
        assert grad_norm > 0.0, f"Gradient nul pour T={T}"


if __name__ == "__main__":
    test_unet_preserves_length_and_backward()
