"""Utilitaires partagés pour l'article C."""

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)
