"""Script CLI pour exécuter le banc QoS par clusters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.scenarios.qos_cluster_bench import run_bench


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Graine de simulation pour l'initialisation du placement et des intervalles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Répertoire de sortie des CSV et du résumé (défaut : results/qos_clusters)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Durée maximale de simulation en secondes pour chaque scénario (défaut : 24h)",
    )
    parser.add_argument(
        "--mixra-solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Force l'utilisation du solveur SciPy (auto) ou du proxy glouton (greedy) pour MixRA-Opt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les impressions de progression",
    )
    return parser


def _progress(current: int, total: int, context: Dict[str, Any]) -> None:
    num_nodes = context.get("num_nodes")
    tx = context.get("packet_interval_s")
    algo = context.get("algorithm")
    print(f"[{current}/{total}] {algo} – N={num_nodes} TX={tx:.0f}s")


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_bench(
        seed=args.seed,
        output_dir=args.output_dir,
        simulation_duration_s=args.duration if args.duration is not None else None,
        mixra_solver=args.mixra_solver,
        quiet=args.quiet,
        progress_callback=None if args.quiet else _progress,
    )
    if not args.quiet:
        report_path = summary.get("report_path")
        summary_path = summary.get("summary_path")
        if report_path:
            print(f"Rapport Markdown : {report_path}")
        if summary_path:
            print(f"Résumé JSON : {summary_path}")
    return summary


if __name__ == "__main__":
    main()
