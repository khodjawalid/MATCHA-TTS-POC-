"""Module de phonémisation, convertit du texte brut en séquences de phonèmes alignables."""
from __future__ import annotations

from typing import List

from phonemizer import phonemize
from phonemizer.separator import Separator


class PhonemizerWrapper:
    """Encapsule phonemizer pour produire une liste de phonèmes IPA à partir de texte brut."""

    def __init__(self, language: str = "en-us", backend: str = "espeak") -> None:
        """Configure le backend espeak-ng pour phonémiser de façon reproductible.

        La phonémisation réduit l'ambiguïté graphème→phonème : le modèle acoustique
        opère sur des unités phonétiques alignables plutôt que sur des caractères bruts,
        ce qui facilite l'apprentissage des correspondances texte→mél dans un cadre TTS.
        """

        self.language = language
        self.backend = backend
        self.punctuation_marks = ";:,.!?¡¿—…“”\"''()[]"

    def __call__(self, text: str) -> List[str]:
        """Renvoie une liste de phonèmes IPA nettoyés et séparés en tokens."""
        separator = Separator(phone=" ", word="|")  # word sep différent pour satisfaire phonemizer
        phoneme_str = phonemize(
            text,
            language=self.language,
            backend=self.backend,
            strip=True,
            punctuation_marks=self.punctuation_marks,
            preserve_punctuation=False,
            with_stress=True,
            njobs=1,
            separator=separator,
        )
        # Nettoyage : remplace le séparateur de mot par espace, coupe et filtre les tokens vides
        phoneme_str = phoneme_str.replace("|", " ")
        tokens = [p.strip() for p in phoneme_str.split(" ") if p.strip()]
        return tokens


def text_to_phonemes(text: str) -> List[str]:
    """Méthode fonctionnelle pour phonémiser rapidement un texte."""
    return PhonemizerWrapper()(text)
