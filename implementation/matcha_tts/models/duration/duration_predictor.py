"""Prédicteur de durées supervisé à partir des alignements MAS."""
from __future__ import annotations

import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """Réseau léger déterministe pour régresser les log-durées par phonème."""

    def __init__(self, channels: int, kernel_size: int, dropout: float, num_layers: int = 2) -> None:
        """Construit une pile de convolutions 1D suivie d'une projection scalaire.

        On prédit log(duree + 1) pour stabiliser l'apprentissage (plage plus compacte)
        et conserver la positivité après exponentiation en inférence.
        """

        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, 1)

    def forward(self, encoded_text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Prédit des log-durées (B, T_text) à partir des représentations encodées.

        Args:
            encoded_text: (B, T_text, D) sorties du TextEncoder.
            text_mask: (B, T_text) bool, True = valide, False = padding.
        Returns:
            log_durations_pred: (B, T_text) valeurs réelles.

        Le modèle est déterministe (pas de sampling) : il fournit un signal d'upsampling
        direct pour expanser les phonèmes vers la timeline mél.
        """

        x = encoded_text.transpose(1, 2)  # (B, D, T)
        x = self.conv(x).transpose(1, 2)  # (B, T, D)
        log_durations = self.proj(x).squeeze(-1)  # (B, T)
        if text_mask is not None:
            padding_mask = ~text_mask
            log_durations = log_durations.masked_fill(padding_mask, 0.0)
        return log_durations

    @torch.no_grad()
    def infer_durations(self, log_durations: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Convertit des log-durées en durées entières minimales (>=1).

        On applique exp, on soustrait 1 (inverse de log(d+1)), puis on arrondit à l'entier
        supérieur. Les positions padding sont forcées à 0.
        """

        durations = torch.exp(log_durations) - 1.0
        durations = torch.ceil(durations).clamp(min=1.0)
        if text_mask is not None:
            padding_mask = ~text_mask
            durations = durations.masked_fill(padding_mask, 0.0)
        return durations


if __name__ == "__main__":
    # Test rapide sur tenseurs factices pour vérifier positivité et shape.
    torch.manual_seed(0)
    B, T, D = 2, 5, 16
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    model = DurationPredictor(channels=D, kernel_size=3, dropout=0.1)
    log_d = model(x, mask)
    d = model.infer_durations(log_d, mask)
    assert log_d.shape == (B, T)
    assert (d >= 0).all()
    print("Shapes ok, durations min:", d.min().item())
