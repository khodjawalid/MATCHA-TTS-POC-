"""
Smoke test du sampler Euler : vérifie que l'intégration ODE tourne et conserve les shapes.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.decoder.unet_1d import UNet1D
from matcha_tts.inference.sampler import euler_sampler


def main():
    torch.manual_seed(0)

    B, n_mels, T = 1, 80, 32
    # mask : True = valide
    mask = torch.ones(B, T, dtype=torch.bool)

    x0 = torch.randn(B, n_mels, T)
    mu = torch.randn(B, n_mels, T)

    model = UNet1D(
        channels=n_mels,
        attention_heads=2,
        time_embedding_dim=256,
        hidden_dim=256,
        num_layers_per_block=1,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout=0.0,
    )
    model.eval()

    out = euler_sampler(unet=model, x0=x0, mu=mu, mel_mask=mask, n_steps=4)

    print("out shape:", out.shape)
    assert out.shape == x0.shape
    assert torch.isfinite(out).all()

    print("✅ Euler sampler smoke test OK")


if __name__ == "__main__":
    main()
