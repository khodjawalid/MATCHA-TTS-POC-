"""Sampler Euler pour résoudre l'ODE du flow en inférence."""
from __future__ import annotations

import torch


def euler_sampler(unet, x0: torch.Tensor, mu: torch.Tensor, mel_mask: torch.Tensor | None = None, n_steps: int = 1) -> torch.Tensor:
    """Intègre l'ODE x' = v_theta(x, t | µ) par Euler explicite.

    Args:
        unet: modèle de vitesse conditionné (UNet 1D).
        x0: bruit initial (B, C, T) gaussien.
        mu: condition acoustique (B, C, T) alignée temporellement.
        mel_mask: (B, T) bool, True = valide (frames à garder).
        n_steps: nombre de pas Euler (NFE).

    Returns:
        x_T: approximation du mél généré après intégration.

    Schéma : x_{i+1} = x_i + h * v_theta(x_i, t_i), avec h = 1/n_steps et t_i = i/n_steps.
    """

    x = x0
    h = 1.0 / float(n_steps)
    for i in range(n_steps):
        t_val = i * h
        t = torch.full((x.size(0),), t_val, device=x.device)
        v = unet(x, t, conditioning=mu, mel_mask=mel_mask)
        if mel_mask is not None:
            valid = mel_mask.float().unsqueeze(1)  # (B, 1, T)
            v = v * valid
        x = x + h * v
    return x


if __name__ == "__main__":
    # Test minimal : vérifie les shapes et absence de NaN avec un UNet factice.
    class DummyUNet(torch.nn.Module):
        def forward(self, x, t, conditioning=None, mel_mask=None):
            return torch.zeros_like(x)

    B, C, T = 2, 4, 6
    x0 = torch.randn(B, C, T)
    mu = torch.randn(B, C, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = euler_sampler(DummyUNet(), x0, mu, mask, n_steps=4)
    assert out.shape == x0.shape
    assert torch.isfinite(out).all()
    print("Euler sampler test ok", out.mean().item())
