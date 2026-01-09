"""
Test unitaire simple du TextEncoder avec Rotary Positional Embeddings (RoPE).

Ce script ne teste pas la qualité linguistique, mais vérifie que :
- le forward fonctionne
- les shapes sont cohérentes
- le masquage du padding est bien pris en compte
- le backward passe sans erreur

À exécuter avant d’implémenter l’alignement (MAS).
"""

import sys
from pathlib import Path

import torch

# Ajoute la racine du projet au PYTHONPATH pour permettre l'import local en mode script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.encoder.text_encoder import TextEncoder


def test_text_encoder_forward():
    print("=== Test du TextEncoder avec RoPE ===")

    # -----------------------------
    # Paramètres factices réalistes
    # -----------------------------
    batch_size = 2
    max_text_len = 16
    vocab_size = 60       # nombre de phonèmes IPA (approx)
    embedding_dim = 128
    num_layers = 2
    num_heads = 4

    # -----------------------------
    # Initialisation du modèle
    # -----------------------------
    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    )

    encoder.train()

    # -----------------------------
    # Batch factice avec padding
    # 0 = padding_idx
    # -----------------------------
    phoneme_ids = torch.tensor([
        [5, 12, 8, 20, 9, 0, 0, 0],
        [4, 7, 13, 6, 11, 10, 2, 1]
    ], dtype=torch.long)

    # TextEncoder attend un masque True sur les positions valides
    phoneme_mask = phoneme_ids != 0

    print("Entrée phoneme_ids shape :", phoneme_ids.shape)
    print("Masque shape             :", phoneme_mask.shape)

    # -----------------------------
    # Forward
    # -----------------------------
    encoded = encoder(phoneme_ids, phoneme_mask)

    print("Sortie encoded shape     :", encoded.shape)

    assert encoded.shape == (batch_size, phoneme_ids.shape[1], embedding_dim), "Shape de sortie incorrecte"

    # -----------------------------
    # Vérification du padding
    # Les positions padding ne doivent pas exploser
    # -----------------------------
    if phoneme_mask.any():
        padding_values = encoded[phoneme_mask]
        print("Valeur max padding :", padding_values.abs().max().item())

    # -----------------------------
    # Test backward (gradient)
    # -----------------------------
    loss = encoded.mean()
    loss.backward()

    grad_norm = 0.0
    for p in encoder.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()

    print("Norme totale des gradients :", grad_norm)
    assert grad_norm > 0, "Aucun gradient détecté"

    print("✅ Test TextEncoder réussi.\n")


if __name__ == "__main__":
    test_text_encoder_forward()
