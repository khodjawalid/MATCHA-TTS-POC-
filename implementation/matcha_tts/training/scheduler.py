"""Construction des schedulers d'apprentissage utilisés pendant l'entraînement."""
from __future__ import annotations

import torch.optim as optim


def build_scheduler(optimizer: optim.Optimizer, config: dict):
    """Crée un scheduler optionnel (ex. warmup ou décroissance) à partir de la config."""
    # TODO: ajouter un scheduler réel (OneCycleLR, CosineAnnealing, etc.)
    return None
