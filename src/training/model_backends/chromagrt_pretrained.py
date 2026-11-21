"""Resolve the checkpoint that a transfer-learning run initialises from.

A finished ChromaGRT run leaves one directory per fold, each holding the
best-validation ``model.pt``. When a whole results root is given as the
transfer source, the fold with the lowest **validation** MAE is selected
automatically, so the choice is never made on the test metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def _fold_validation_mae(fold_dir: Path) -> float:
    """Lowest validation MAE logged for one fold.

    Lightning's CSV logger writes one ``val_mae`` row per validation pass; the
    lowest of them is the epoch the fold's ``model.pt`` was taken from.
    """
    best = float("inf")
    for metrics_csv in sorted(fold_dir.glob("lightning_logs/version_*/metrics.csv")):
        with metrics_csv.open() as handle:
            for row in csv.DictReader(handle):
                value = row.get("val_mae")
                if value:
                    best = min(best, float(value))
    return best


def resolve_pretrained_checkpoint(path: Path) -> Path:
    """Return the ``model.pt`` to initialise a transfer-learning run from.

    ``path`` is either a checkpoint file, or a results root holding one
    ``fold_N/<timestamp>/model.pt`` per fold. In the second case the fold with
    the lowest validation MAE wins.
    """
    path = Path(path)
    if path.is_file():
        return path
    candidates = []
    for checkpoint in sorted(path.glob("fold_*/**/model.pt")):
        candidates.append((_fold_validation_mae(checkpoint.parent), checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No model.pt checkpoint found under {path}")
    return min(candidates)[1]


def load_pretrained_state_dict(path: Path) -> dict:
    """Load a bare ``state_dict`` checkpoint written by ``torch.save``."""
    return torch.load(path, map_location="cpu")
