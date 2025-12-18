"""Exécution rapide SNIR on/off pour valider les figures combinées."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.aggregate_step1_results import aggregate_step1_results
from scripts.plot_step1_results import (
    DEFAULT_FIGURES_DIR,
    generate_step1_figures,
)
from scripts.run_step1_experiments import main as run_step1_experiment

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "snir_validation"


def _run_snir_experiments(
    results_dir: Path,
    nodes: int,
    packet_interval: float,
    duration: float,
    seed: int,
    algorithm: str,
    states: Iterable[bool],
) -> List[Path]:
    csv_paths: List[Path] = []
    for use_snir in states:
        argv = [
            f"--nodes={nodes}",
            f"--packet-interval={packet_interval}",
            f"--duration={duration}",
            f"--seed={seed}",
            f"--algorithm={algorithm}",
            f"--output-dir={results_dir}",
            "--quiet",
        ]
        if use_snir:
            argv.append("--use-snir")

        result = run_step1_experiment(argv)
        csv_path = Path(result.get("csv_path"))
        csv_paths.append(csv_path)
        print(f"[OK] Simulation SNIR={'on' if use_snir else 'off'} : {csv_path.relative_to(ROOT_DIR)}")
    return csv_paths


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
    compare_plots = list(figures_dir.glob("*_snir_compare_*.png"))
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
    parser.add_argument("--nodes", type=int, default=20, help="Nombre de nœuds pour le test rapide")
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=120.0,
        help="Intervalle moyen d'émission (secondes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
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

    _run_snir_experiments(
        args.results_dir,
        args.nodes,
        args.packet_interval,
        args.duration,
        args.seed,
        args.algorithm,
        states=(False, True),
    )
    _aggregate_results(args.results_dir)

    figures_root = _generate_plots(args.results_dir, args.figures_dir)
    _assert_compare_plots(figures_root)


if __name__ == "__main__":
    main()
