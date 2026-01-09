"""
Vérifie que le mapping phonèmes→ids reste dans les bornes du vocab statique
et gère les OOV sans dépasser vocab_size.
Affiche aussi un compteur brut vs normalisé pour inspection rapide.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

from matcha_tts.data.phonemizer import PhonemizerWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.text.symbols import SYMBOLS, UNK_ID, ID_TO_SYMBOL
from matcha_tts.text.text_to_sequence import phonemes_to_ids, normalize_ipa_tokens


def test_phonemes_to_ids_bounds():
    tokens = ["p", "a", "t", "ɹ", "i", "d͡ʒ", "!", "<oov_token>"]
    ids = phonemes_to_ids(tokens)

    assert len(ids) == len(tokens)
    assert max(ids) < len(SYMBOLS)
    assert ids[-1] == UNK_ID


def demo_counts():
    ph = PhonemizerWrapper()

    raw_counts = Counter()
    norm_counts = Counter()
    unk_counts = 0

    texts = [
        "Hello world",
        "This is a test sentence",
        "The quick brown fox jumps over the lazy dog",
    ]

    for text in texts:
        raw = ph(text)
        norm = normalize_ipa_tokens(raw)

        raw_counts.update(raw)
        norm_counts.update(norm)

        ids = phonemes_to_ids(raw)  # phonemes_to_ids normalise déjà en interne si tu l’appelles sur raw
        unk_counts += sum(1 for i in ids if i == UNK_ID)

    print("\n--- RAW tokens (phonemizer output) ---")
    print(raw_counts)

    print("\n--- NORMALIZED tokens (after segmentation) ---")
    print(norm_counts)

    print(f"\nUNK count (after mapping): {unk_counts}")


if __name__ == "__main__":
    test_phonemes_to_ids_bounds()
    print("✅ vocab id test passed")
    demo_counts()
