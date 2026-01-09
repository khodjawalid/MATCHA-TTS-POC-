"""Utilitaire de logging centralisé pour harmoniser les messages dans tout le projet."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Configure et retourne un logger avec sortie console et option fichier."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
