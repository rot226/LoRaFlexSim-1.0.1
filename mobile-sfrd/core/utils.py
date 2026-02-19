"""Utilitaires transverses pour mobile-sfrd."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Charge un fichier YAML et retourne son contenu.

    Args:
        path: Chemin vers le fichier YAML.

    Returns:
        Le contenu du YAML sous forme de dictionnaire.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def path_join(*parts: str) -> str:
    """Assemble des fragments de chemin de façon portable."""
    return os.path.join(*parts)


def ensure_output_dirs(base_dir: str) -> Dict[str, str]:
    """Crée les répertoires de sortie standards et retourne leurs chemins."""
    outputs_dir = path_join(base_dir, "outputs")
    csv_dir = path_join(outputs_dir, "csv")
    figures_dir = path_join(outputs_dir, "figures")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return {
        "outputs": outputs_dir,
        "csv": csv_dir,
        "figures": figures_dir,
    }


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde un DataFrame en CSV avec configuration standardisée."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(path, index=False)
