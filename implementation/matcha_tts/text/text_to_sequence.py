"""Conversions texte/phonèmes → IDs à partir d'un vocab fixe.

Aucun ID n'est créé dynamiquement : les OOV sont mappés sur <unk> (ou ignorés)
pour garantir un embedding stable et reproductible. Les tokens IPA potentiellement
combinés (stress collé, longueur en suffixe) sont segmentés en symboles atomiques
avant mapping pour rester conformes au vocabulaire fixe.
"""
from __future__ import annotations

import warnings
from typing import List

from matcha_tts.text.symbols import SYMBOL_TO_ID, UNK_ID, UNK_SYMBOL


def normalize_ipa_tokens(tokens: List[str]) -> List[str]:
    """Segmente les tokens IPA combinés (stress/longueur) en symboles atomiques.

    Règles simples (pédagogiques) :
    1) Stress collé en préfixe (ˈ, ˌ) → extrait en token séparé puis traite la base.
    2) Longueur en suffixe (ː, ˑ) → extrait en token séparé après la base.
    3) Diphtongues connues (eɪ, aɪ, aʊ, ɔɪ, oʊ) ne sont pas cassées.
    4) Tokens vides sont ignorés.
    """

    stress_markers = {"ˈ", "ˌ"}
    length_markers = {"ː", "ˑ"}
    diphthongs = {"eɪ", "aɪ", "aʊ", "ɔɪ", "oʊ"}

    normalized: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # 1) extrait stress en préfixe (y compris cas avec double stress éventuel)
        while tok and tok[0] in stress_markers:
            normalized.append(tok[0])
            tok = tok[1:]

        if not tok:
            continue

        # 2) diphtongues connues : on les garde intactes (avant de gérer longueur)
        base = tok

        # 3) longueur en suffixe (on tolère un seul marqueur, IPA simplifiée)
        length_suffix = None
        if base and base[-1] in length_markers:
            length_suffix = base[-1]
            base = base[:-1]

        if not base:
            continue

        # Conserve diphtongue si dans la liste, sinon base telle quelle
        normalized.append(base)

        # Ajoute le marqueur de longueur après la base si présent
        if length_suffix is not None:
            normalized.append(length_suffix)

    return normalized


def phonemes_to_ids(phoneme_tokens: List[str]) -> List[int]:
    """Mappe des tokens phonémiques vers des IDs dans le vocabulaire statique.

    Normalise d'abord les tokens IPA combinés (stress/longueur), puis applique le mapping.
    - tokens vides sont ignorés
    - tokens OOV sont mappés vers <unk> s'il existe ; sinon ignorés avec warning
    """

    ids: List[int] = []
    for tok in normalize_ipa_tokens(phoneme_tokens):
        if tok in SYMBOL_TO_ID:
            ids.append(SYMBOL_TO_ID[tok])
        else:
            if UNK_SYMBOL in SYMBOL_TO_ID:
                ids.append(UNK_ID)
            else:
                warnings.warn(f"Phoneme token '{tok}' is OOV and <unk> is not defined; token ignored.")
    return ids


__all__ = ["phonemes_to_ids", "normalize_ipa_tokens"]
