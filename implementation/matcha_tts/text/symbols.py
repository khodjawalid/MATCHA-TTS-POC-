"""Vocabulaire phonémique statique pour Matcha-TTS.

On évite tout vocabulaire dynamique pour garantir la reproductibilité et
rester fidèle aux implémentations Matcha-TTS/Grad-TTS : un embedding fixe,
indexé de façon déterministe, supprime les IndexError et stabilise
l'entraînement.
"""
from __future__ import annotations

from typing import Dict, List

# Symboles spéciaux
PAD_SYMBOL = "_"  # padding explicite
UNK_SYMBOL = "<unk>"  # pour les OOV

# Ponctuation et espace conservés (utile pour les pauses et prosodie)
PUNCTUATION = list(";:,.!?—…\"«»“”[]() ")

# Inventaire IPA couvrant les phonèmes courants de LJSpeech (anglais) via espeak-ng.
# On privilégie un set large et statique pour éviter les OOV ; tout phonème manquant
# sera replié sur <unk> mais jamais ajouté dynamiquement.
IPA_PHONEMES = [
    # --------------------
    # Occlusives (plosives)
    # --------------------
    "p", "b", "t", "d", "k", "g", "ɡ", "ʔ",

    # --------------------
    # Affriquées
    # --------------------
    "t͡ʃ", "d͡ʒ",   # ligature
    "tʃ", "dʒ",     # sans ligature (espeak-ng peut produire les deux)

    # --------------------
    # Nasales
    # --------------------
    "m", "n", "ŋ",

    # --------------------
    # Fricatives
    # --------------------
    "f", "v", "θ", "ð",
    "s", "z", "ʃ", "ʒ",
    "h",

    # --------------------
    # Approximantes / liquides
    # --------------------
    "l", "ɫ",       # l clair / l sombre
    "r", "ɹ",       # variantes r
    "j", "w",

    # --------------------
    # Flap / allophones
    # --------------------
    "ɾ",

    # --------------------
    # Voyelles monophthongues
    # --------------------
    "i", "ɪ",
    "e", "ɛ",
    "æ",
    "a", "ɑ", "ɐ",
    "ɒ",
    "ɔ",
    "o",
    "ʊ", "u",
    "ʌ",
    "ə",
    "ɚ", "ɝ",

    # --------------------
    # Diphtongues courantes
    # --------------------
    "eɪ",
    "aɪ",
    "aʊ",
    "ɔɪ",
    "oʊ",

    # --------------------
    # Longueur / prosodie
    # --------------------
    "ˈ",    # stress primaire
    "ˌ",    # stress secondaire
    "ː",    # voyelle longue
    "ˑ",    # semi-longue (rare mais possible)

    # --------------------
    # Silences / pauses
    # --------------------
    "‖",    # pause forte (rare)
    "|",    # pause faible (si jamais présent)
]


# Liste finale : ordre déterministe (PAD puis UNK puis phonèmes puis ponctuation)
SYMBOLS: List[str] = [PAD_SYMBOL, UNK_SYMBOL] + IPA_PHONEMES + PUNCTUATION

SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL: Dict[int, str] = {i: s for s, i in SYMBOL_TO_ID.items()}

# Identifiants pratiques
PAD_ID = SYMBOL_TO_ID[PAD_SYMBOL]
UNK_ID = SYMBOL_TO_ID[UNK_SYMBOL]

__all__ = [
    "SYMBOLS",
    "SYMBOL_TO_ID",
    "ID_TO_SYMBOL",
    "PAD_SYMBOL",
    "UNK_SYMBOL",
    "PAD_ID",
    "UNK_ID",
]
