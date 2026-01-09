"""Script d'entraînement end-to-end Matcha-TTS (pédagogique)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Assure que le dossier racine du projet est dans le PYTHONPATH quand on lance via `python -m matcha_tts.train`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.data.dataset import LJSpeechDataset, collate_fn
from matcha_tts.data.phonemizer import PhonemizerWrapper
from matcha_tts.models.matcha_tts import MatchaTTS
from matcha_tts.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Arguments CLI pour configurer l'entraînement et les overrides."""
    parser = argparse.ArgumentParser(description="Train Matcha-TTS")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Fichier de configuration YAML")
    parser.add_argument("--dataset_path", type=Path, default=None, help="Racine du dataset (contient metadata.csv et wavs/)")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Répertoire de sortie (checkpoints/logs)")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    parser.add_argument("--max_steps", type=int, default=None, help="Override du nombre maximal de steps")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint à reprendre")
    parser.add_argument("--overfit_one_batch", action="store_true", help="Debug : sur-apprendre un seul batch fixe")
    parser.add_argument("--dry_run_steps", type=int, default=0, help="Limite rapide du nombre de steps pour un dry-run")
    parser.add_argument("--num_workers", type=int, default=None, help="Override du nombre de workers DataLoader")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None, help="Active/désactive pin_memory du DataLoader")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=None, help="Active/désactive persistent_workers")
    parser.add_argument("--val_every", type=int, default=None, help="Fréquence de validation (override validation.every_steps)")
    parser.add_argument("--no_val", action="store_true", help="Désactive la validation périodique")
    parser.add_argument("--overfit", action="store_true", help="Active le mode overfit_one_batch")
    parser.add_argument("--overfit_steps", type=int, default=None, help="Nombre de steps en mode overfit")
    parser.add_argument("--save_every", type=int, default=None, help="Fréquence de sauvegarde des checkpoints (override ckpt_every)")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloader(
    config: dict,
    dataset_path: Path,
    phonemizer: PhonemizerWrapper,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
) -> DataLoader:
    """Construit le DataLoader LJSpeech avec collate_fn et masques True=valide.

    En environnement serveur/Jupyter, on privilégie num_workers=0 pour éviter les soucis
    de multiprocessing et simplifier le debug.
    """

    audio_cfg = config["audio"]

    metadata_path = dataset_path / "metadata.csv"
    dataset = LJSpeechDataset(metadata_path=metadata_path, audio_dir=dataset_path, phonemizer=phonemizer, audio_config=audio_cfg)

    train_cfg = config["training"]
    nw = num_workers if num_workers is not None else train_cfg.get("num_workers", 0)
    pm = pin_memory if pin_memory is not None else train_cfg.get("pin_memory", False)
    pw = persistent_workers if persistent_workers is not None else train_cfg.get("persistent_workers", False)

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=pw and nw > 0,
        collate_fn=collate_fn,
    )
    return loader


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.dry_run_steps and args.dry_run_steps > 0:
        config.setdefault("training", {})["max_steps"] = min(
            config["training"].get("max_steps", args.dry_run_steps), args.dry_run_steps
        )
    if args.save_every is not None:
        config.setdefault("training", {})["ckpt_every"] = args.save_every
    if args.val_every is not None:
        config.setdefault("validation", {})["every_steps"] = args.val_every
    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = args.num_workers
    if args.pin_memory is not None:
        config.setdefault("training", {})["pin_memory"] = args.pin_memory
    if args.persistent_workers is not None:
        config.setdefault("training", {})["persistent_workers"] = args.persistent_workers
    if args.overfit or args.overfit_one_batch:
        config.setdefault("training", {})["overfit_one_batch"] = True
    if args.overfit_steps is not None:
        config.setdefault("training", {})["overfit_steps"] = args.overfit_steps
    if args.overfit and not args.dry_run_steps:
        config.setdefault("training", {})["max_steps"] = config["training"].get("overfit_steps", 200)

    dataset_root = args.dataset_path or Path(config["paths"]["data_root"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str in (None, "auto"):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    phon_cfg = config.get("phonemizer", {})
    phonemizer = PhonemizerWrapper(
        language=phon_cfg.get("language", "en-us"),
        backend=phon_cfg.get("backend", "espeak"),
    )

    # Données
    train_loader = build_dataloader(
        config,
        dataset_root,
        phonemizer,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    # Modèle
    model = MatchaTTS(config)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        overfit_one_batch=args.overfit_one_batch or args.overfit,
        output_dir=output_dir,
        phonemizer=phonemizer,
        validation_texts=config.get("validation", {}).get("texts", []),
        val_every=args.val_every if args.val_every is not None else config.get("validation", {}).get("every_steps"),
        enable_validation=not args.no_val,
    )

    if args.resume is not None:
        last_step = trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at step {last_step}")

    # Boucle principale (sans validation pour simplicité pédagogique)
    while trainer.global_step < trainer.max_steps:
        trainer.train_one_epoch()
        if trainer.overfit_one_batch:
            # En mode overfit, on peut s'arrêter dès que max_steps est atteint
            break


if __name__ == "__main__":
    main()
