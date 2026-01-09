"""Script d'inférence Matcha-TTS : texte → phonèmes → mél (sans vocoder)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Ajout de la racine projet pour exécution directe
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.matcha_tts import MatchaTTS
from matcha_tts.data.phonemizer import PhonemizerWrapper


def parse_args() -> argparse.Namespace:
    """Construit les arguments de ligne de commande pour lancer l'inférence."""
    parser = argparse.ArgumentParser(description="Inference for Matcha-TTS")
    parser.add_argument("--config", type=Path, required=True, help="Chemin vers le fichier de configuration YAML")
    parser.add_argument("--text", type=str, required=True, help="Texte à synthétiser")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint du modèle Matcha-TTS")
    parser.add_argument("--output", type=Path, default=Path("outputs/mel.pt"), help="Fichier de sortie pour le mél généré (.pt ou .npy)")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    # Résout le chemin par rapport à la racine du projet si besoin
    if not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate
    with path.open("r") as f:
        return yaml.safe_load(f)


def phonemes_to_ids(phonemes, vocab_size: int) -> torch.Tensor:
    """Mappe la liste de phonèmes vers des IDs simples (pad=0, unk=1)."""
    mapping = {"<pad>": 0, "<unk>": 1}
    ids = []
    for p in phonemes:
        if p not in mapping:
            if len(mapping) < vocab_size:
                mapping[p] = len(mapping)
            else:
                mapping[p] = 1  # unk
        ids.append(mapping[p])
    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    """Point d'entrée : charge config/ckpt, phonémise le texte et génère un mél."""
    args = parse_args()
    cfg = load_config(args.config)

    model = MatchaTTS(cfg)
    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[Warn] State dict mismatch", missing, unexpected)
    else:
        print("[Warn] Aucun checkpoint fourni, inférence avec poids aléatoires.")

    model.eval()

    phonemizer = PhonemizerWrapper()
    phonemes = phonemizer(args.text)
    phoneme_ids = phonemes_to_ids(phonemes, vocab_size=cfg["model"]["vocab_size"])
    phoneme_ids = phoneme_ids.unsqueeze(0)  # batch 1
    phoneme_mask = phoneme_ids == 0

    n_steps = cfg.get("inference", {}).get("n_steps", 30)
    temperature = cfg.get("inference", {}).get("temperature", 1.0)

    with torch.no_grad():
        mel, mel_mask = model.infer(phoneme_ids, phoneme_mask, n_steps=n_steps, temperature=temperature)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mel": mel.cpu(), "mel_mask": mel_mask.cpu()}, args.output)
    print(f"Mél généré sauvegardé dans {args.output} | shape mel: {tuple(mel.shape)} | mask shape: {tuple(mel_mask.shape)}")


if __name__ == "__main__":
    main()
