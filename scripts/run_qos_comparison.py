"""Script CLI pour comparer un scénario QoS et une ligne de base ADR désactivée."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.scenarios.qos_comparison import run_qos_vs_adr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=24, help="Nombre de nœuds simulés")
    parser.add_argument(
        "--packets-per-node",
        type=int,
        default=6,
        help="Nombre de paquets émis par nœud avant arrêt",
    )
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=180.0,
        help="Intervalle moyen d'émission (secondes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Durée maximale de simulation en secondes (None = jusqu'à épuisement)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Graine de simulation")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dossier de sortie pour les journaux et métriques",
    )
    parser.add_argument("--quiet", action="store_true", help="Silence les impressions de progression")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_qos_vs_adr(
        num_nodes=args.nodes,
        packets_per_node=args.packets_per_node,
        packet_interval=args.packet_interval,
        duration_s=args.duration,
        seed=args.seed,
        output_dir=args.output,
        quiet=args.quiet,
    )
    if not args.quiet:
        summary_path = summary["summary_path"]
        print(f"Résumé écrit dans {summary_path}")
    return summary


if __name__ == "__main__":
    main()
