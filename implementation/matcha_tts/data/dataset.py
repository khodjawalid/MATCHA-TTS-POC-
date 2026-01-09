"""Implémentation du dataset LJSpeech, responsable du chargement des paires texte/audio et de la préparation des entrées modèle."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .phonemizer import PhonemizerWrapper
from .preprocessing import compute_mel_spectrogram, load_audio, load_metadata, normalize_mel
from ..utils.mask import create_padding_mask
from matcha_tts.text.symbols import PAD_ID, SYMBOLS
from matcha_tts.text.text_to_sequence import phonemes_to_ids


class LJSpeechDataset(Dataset):
    """Dataset minimal pour LJSpeech, prépare texte phonémisé et mél-spectrogramme normalisé."""

    def __init__(
        self,
        metadata_path: Path,
        audio_dir: Path,
        phonemizer: PhonemizerWrapper,
        audio_config: Dict[str, int],
    ) -> None:
        """Charge les métadonnées et configure les hyperparamètres audio.

        Pas d'alignements externes : la sortie se limite à (phonèmes, mél) et leurs longueurs,
        ce qui servira de base aux modules de durée/alignement plus tard.
        """

        super().__init__()
        self.audio_dir = audio_dir
        self.phonemizer = phonemizer
        self.audio_config = audio_config
        self.samples = self._load_metadata(metadata_path)

        self.pad_id = PAD_ID
        self.vocab_size = len(SYMBOLS)

    def _load_metadata(self, metadata_path: Path) -> List[Tuple[Path, str]]:
        """Lit metadata.csv et associe chaque ID au chemin wav et au texte normalisé."""
        pairs = []
        for utt_id, text in load_metadata(metadata_path):
            audio_path = self.audio_dir / "wavs" / f"{utt_id}.wav"
            pairs.append((audio_path, text))
        return pairs

    def __len__(self) -> int:
        """Retourne le nombre d'échantillons disponibles."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Prépare un élément : phonèmes, longueurs et mél normalisé."""
        audio_path, text = self.samples[index]

        phonemes = self.phonemizer(text)
        phoneme_ids = torch.tensor(phonemes_to_ids(phonemes), dtype=torch.long)

        waveform = load_audio(audio_path, self.audio_config["sample_rate"])
        mel = compute_mel_spectrogram(
            waveform=waveform,
            sample_rate=self.audio_config["sample_rate"],
            n_fft=self.audio_config["n_fft"],
            hop_length=self.audio_config["hop_length"],
            win_length=self.audio_config["win_length"],
            n_mels=self.audio_config["n_mels"],
            fmin=self.audio_config.get("fmin", 0),
            fmax=self.audio_config.get("fmax", 8000),
        )
        mel = normalize_mel(mel)

        return {
            "phonemes": phoneme_ids,
            "phoneme_length": torch.tensor(len(phoneme_ids), dtype=torch.long),
            "mel": mel,
            "mel_length": torch.tensor(mel.shape[-1], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int = PAD_ID) -> Dict[str, torch.Tensor]:
    """Assemble un lot en paddant phonèmes/mels et en générant des masques True=valide.

    Le lien texte ↔ frames est conservé via les longueurs : les phonèmes et le mél
    sont alignés implicitement par durée (à apprendre plus tard via MAS/durations).
    """

    phoneme_lengths = torch.tensor([item["phoneme_length"] for item in batch], dtype=torch.long)
    mel_lengths = torch.tensor([item["mel_length"] for item in batch], dtype=torch.long)

    max_ph = int(phoneme_lengths.max().item())
    max_mel = int(mel_lengths.max().item())
    n_mels = batch[0]["mel"].shape[0]

    phonemes_padded = torch.full((len(batch), max_ph), pad_id, dtype=torch.long)
    mels_padded = torch.zeros((len(batch), n_mels, max_mel), dtype=torch.float)

    for i, item in enumerate(batch):
        ph_len = item["phoneme_length"].item()
        mel_len = item["mel_length"].item()
        phonemes_padded[i, :ph_len] = item["phonemes"]
        mels_padded[i, :, :mel_len] = item["mel"]

    phoneme_padding_mask = create_padding_mask(phoneme_lengths, max_ph)
    mel_padding_mask = create_padding_mask(mel_lengths, max_mel)

    phoneme_mask = ~phoneme_padding_mask  # True = valide
    mel_mask = ~mel_padding_mask          # True = valide

    return {
        "phonemes": phonemes_padded,
        "phoneme_length": phoneme_lengths,
        "phoneme_mask": phoneme_mask,
        "phoneme_padding_mask": phoneme_padding_mask,
        "mel": mels_padded,
        "mel_length": mel_lengths,
        "mel_mask": mel_mask,
        "mel_padding_mask": mel_padding_mask,
    }
