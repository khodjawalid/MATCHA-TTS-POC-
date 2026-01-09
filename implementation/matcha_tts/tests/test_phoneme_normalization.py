"""
Tests de normalisation/segmentation des tokens IPA combinés et mapping vers IDs.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.text.symbols import SYMBOLS
from matcha_tts.text.text_to_sequence import normalize_ipa_tokens, phonemes_to_ids


def test_normalize_stress_and_length():
    assert normalize_ipa_tokens(["ˈɑː"]) == ["ˈ", "ɑ", "ː"]
    assert normalize_ipa_tokens(["ˌoʊ"]) == ["ˌ", "oʊ"]
    assert normalize_ipa_tokens(["ˈeɪ"]) == ["ˈ", "eɪ"]
    assert normalize_ipa_tokens(["ɑː"]) == ["ɑ", "ː"]
    assert normalize_ipa_tokens(["ɐ"]) == ["ɐ"]


def test_ids_within_vocab():
    tokens = ["ˈɑː", "ˌoʊ", "ɐ"]
    ids = phonemes_to_ids(tokens)
    assert ids, "IDs should not be empty"
    assert max(ids) < len(SYMBOLS)


if __name__ == "__main__":
    test_normalize_stress_and_length()
    test_ids_within_vocab()
    print("✅ phoneme normalization tests passed")
