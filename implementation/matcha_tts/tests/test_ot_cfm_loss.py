"""
Smoke test OT-CFM : vérifie que la loss est finie et rétropropagable avec le UNet réel.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.decoder.unet_1d import UNet1D
from matcha_tts.training.losses import conditional_flow_matching_loss


def main():
    torch.manual_seed(0)

    B, n_mels, T = 2, 80, 64
    # mask (True = valide)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, -20:] = False

    x1 = torch.randn(B, n_mels, T)         # mel "réel" factice
    mu = torch.randn(B, n_mels, T)         # condition factice

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

    loss = conditional_flow_matching_loss(unet=model, mel=x1, mu=mu, mel_mask=mask)

    print("loss:", loss.detach().item())
    assert torch.isfinite(loss), "Loss NaN/Inf"
    loss.backward()

    # Vérifie qu'au moins un param a un gradient non nul
    grad_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += p.grad.abs().sum().item()
    print("grad sum:", grad_sum)
    assert grad_sum > 0, "Pas de gradients sur le modèle"

    print("✅ OT-CFM loss smoke test OK")


if __name__ == "__main__":
    main()
