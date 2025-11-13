#!/usr/bin/env python3
"""Script utilitaire pour générer et convertir les figures QoS.

Prérequis :
- le script ``extract_metrics.sh`` doit avoir été exécuté préalablement afin de
  préparer les métriques nécessaires ;
- le script ne lance pas automatiquement de simulations, il exploite les
  résultats déjà présents dans ``results/``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génère les figures QoS et les convertit en JPG/EPS.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écrase les fichiers de sortie existants (PNG/JPG/EPS) si nécessaire.",
    )
    return parser.parse_args()


def run_lfs_plots() -> None:
    command = [
        sys.executable,
        "qos_cli/lfs_plots.py",
        "--in",
        "results/",
        "--config",
        "qos_cli/scenarios_small.yaml",
        "--out",
        "qos_cli/figures/",
    ]

    LOGGER.info("Exécution de la commande : %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Échec de l'exécution de {' '.join(command)} (code {result.returncode}).")


def convert_figures(overwrite: bool) -> None:
    figures_dir = Path("qos_cli/figures/")
    if not figures_dir.exists():
        raise FileNotFoundError(
            "Le dossier des figures 'qos_cli/figures/' est introuvable. Vérifiez l'exécution de la commande lfs_plots."
        )

    png_files = sorted(figures_dir.glob("*.png"))
    if not png_files:
        LOGGER.warning("Aucun fichier PNG trouvé dans %s.", figures_dir)

    for png_file in png_files:
        _convert_single(png_file, overwrite)


def _convert_single(png_path: Path, overwrite: bool) -> None:
    if not png_path.exists():
        LOGGER.warning("Le fichier %s n'existe pas (ignoré).", png_path)
        return

    targets = {
        "JPEG": png_path.with_suffix(".jpg"),
        "EPS": png_path.with_suffix(".eps"),
    }

    for fmt, target in targets.items():
        if target.exists() and not overwrite:
            LOGGER.info("Fichier %s déjà présent, conversion %s ignorée.", target, fmt)
            continue

        LOGGER.info("Conversion de %s vers %s (%s).", png_path, target, fmt)
        with Image.open(png_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(target, format=fmt)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    run_lfs_plots()
    convert_figures(args.overwrite)


if __name__ == "__main__":
    main()
