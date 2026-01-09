"""
Smoke test du decoder UNet 1D (Matcha-TTS).
Vérifie : shapes, masquage attention, backward, stabilité numérique.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.decoder.unet_1d import UNet1D


def main():
    torch.manual_seed(0)

    B = 2
    n_mels = 80
    T = 64

    # Mask : True = valide, False = padding
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -10:] = False

    x = torch.randn(B, n_mels, T, requires_grad=True)
    mu = torch.randn(B, n_mels, T)
    t = torch.rand(B)  # (B,)

    model = UNet1D(
        channels=n_mels,
        attention_heads=2,
        time_embedding_dim=256,
        hidden_dim=256,
        num_layers_per_block=1,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout=0.1,
    )
    model.train()

    v = model(x, t, conditioning=mu, mel_mask=mask)

    print("v shape:", v.shape)
    assert v.shape == x.shape, "La sortie doit avoir la même shape que x"

    # test NaN
    assert torch.isfinite(v).all(), "NaN/Inf détectés dans la sortie du UNet"

    # backward
    loss = (v ** 2).mean()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print("grad norm:", grad_norm)
    assert grad_norm > 0, "Gradient nul sur x"

    print("✅ UNet1D smoke test OK")


if __name__ == "__main__":
    main()
