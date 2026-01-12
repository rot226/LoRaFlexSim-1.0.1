"""Campagne Step1 dédiée à la comparaison des fenêtres SNIR.

Ce script exécute la matrice Step1 pour plusieurs modes de fenêtre
("packet", "preamble", "symbol") afin de comparer l'impact sur le SNIR.
Les résultats sont rangés dans ``results/step1/windows/`` par défaut.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import run_step1_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1" / "windows"
DEFAULT_WINDOWS: Sequence[str] = ("packet", "preamble", "symbol")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help="Fenêtres SNIR à comparer (packet, preamble, symbol).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire racine pour les CSV générés.",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(run_step1_matrix.DEFAULT_ALGOS),
        help="Algorithmes à tester.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(run_step1_matrix.DEFAULT_SEEDS),
        help="Graines utilisées pour la simulation.",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=list(run_step1_matrix.DEFAULT_NODE_COUNTS),
        help="Charges en nombre de nœuds.",
    )
    parser.add_argument(
        "--packet-intervals",
        nargs="+",
        type=float,
        default=list(run_step1_matrix.DEFAULT_PACKET_INTERVALS),
        help="Périodes moyennes d'émission (secondes).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=run_step1_matrix.DEFAULT_DURATION,
        help="Durée maximale des simulations (secondes).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ignore les combinaisons dont le CSV de sortie existe déjà.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_step1_matrix.main(
        [
            "--algos",
            *args.algos,
            "--with-snir",
            "true",
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--nodes",
            *(str(node) for node in args.nodes),
            "--packet-intervals",
            *(str(interval) for interval in args.packet_intervals),
            "--duration",
            str(args.duration),
            "--snir-windows",
            *args.windows,
            "--results-dir",
            str(args.results_dir),
            *(
                ["--skip-existing"]
                if args.skip_existing
                else []
            ),
        ]
    )


if __name__ == "__main__":
    main()
