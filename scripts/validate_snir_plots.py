"""Exécution rapide SNIR on/off via la matrice pour valider les figures combinées."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.aggregate_step1_results import aggregate_step1_results
from scripts.plot_step1_results import (
    DEFAULT_FIGURES_DIR,
    generate_step1_figures,
)
from scripts.run_step1_matrix import main as run_step1_matrix

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "snir_validation"


def _run_snir_matrix(
    results_dir: Path,
    nodes: int,
    packet_interval: float,
    duration: float,
    seed: int,
    algorithm: str,
) -> None:
    argv = [
        "--algos",
        algorithm,
        "--with-snir",
        "false",
        "true",
        "--seeds",
        str(seed),
        "--nodes",
        str(nodes),
        "--packet-intervals",
        str(packet_interval),
        "--duration",
        str(duration),
        "--results-dir",
        str(results_dir),
    ]
    print("[RUN] Matrice SNIR on/off minimale via run_step1_matrix.py …")
    run_step1_matrix(argv)
    generated = sorted(results_dir.rglob("*_snir-*.csv"))
    if generated:
        for path in generated:
            print(f"[OK] CSV généré : {path.relative_to(ROOT_DIR)}")
    else:
        print("[WARN] Aucun CSV SNIR détecté après l'exécution ; vérifiez les paramètres.")


def _aggregate_results(results_dir: Path) -> None:
    print("[RUN] Agrégation des CSV …")
    aggregate_step1_results(results_dir, strict_snir_detection=True, split_snir=False)


def _generate_plots(results_dir: Path, figures_dir: Path) -> Path:
    print("[RUN] Génération des figures …")
    generate_step1_figures(
        results_dir,
        figures_dir,
        use_summary=True,
        plot_cdf=False,
        compare_snir=True,
    )
    return figures_dir / "step1"


def _assert_compare_plots(figures_dir: Path) -> None:
    compare_plots = list(figures_dir.glob("*_snir-compare_*.png"))
    if not compare_plots:
        raise RuntimeError(
            f"Aucune figure SNIR combinée trouvée dans {figures_dir};"
            " vérifiez l'agrégation ou le tracé."
        )
    print(f"[OK] {len(compare_plots)} figure(s) SNIR combinées détectées dans {figures_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire de sortie pour les CSV SNIR on/off",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Répertoire racine pour les figures générées",
    )
    parser.add_argument("--nodes", type=int, default=10, help="Nombre de nœuds pour le test rapide")
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=60.0,
        help="Intervalle moyen d'émission (secondes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Durée maximale de simulation (secondes)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine de simulation")
    parser.add_argument(
        "--algorithm",
        choices=["adr", "apra", "mixra_h", "mixra_opt"],
        default="adr",
        help="Algorithme QoS à tester",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    _run_snir_matrix(
        args.results_dir,
        args.nodes,
        args.packet_interval,
        args.duration,
        args.seed,
        args.algorithm,
    )
    _aggregate_results(args.results_dir)

    figures_root = _generate_plots(args.results_dir, args.figures_dir)
    _assert_compare_plots(figures_root)


if __name__ == "__main__":
    main()
