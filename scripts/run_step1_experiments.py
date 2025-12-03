"""Automatise l'exécution des expériences de l'étape 1.

Le script permet de choisir l'algorithme QoS (ADR, APRA, MixRA-H ou
MixRA-Opt), de contrôler l'utilisation du calcul SNIR et d'exporter les
métriques clés au format CSV via l'infrastructure de journalisation
existante.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.qos import QoSManager
from loraflexsim.launcher.simulator import InterferenceTracker
from loraflexsim.scenarios.qos_cluster_bench import (
    PAYLOAD_BYTES,
    _apply_adr_pure,
    _apply_apra_like,
    _apply_mixra_h,
    _apply_mixra_opt,
    _compute_additional_metrics,
    _create_simulator,
    _flatten_metrics,
    _write_csv,
    _configure_clusters,
)

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
ALGORITHMS: Mapping[str, Callable[..., None]] = {
    "adr": _apply_adr_pure,
    "apra": _apply_apra_like,
    "mixra_h": _apply_mixra_h,
    "mixra_opt": _apply_mixra_opt,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        default="adr",
        help="Algorithme testé (adr, apra, mixra_h, mixra_opt)",
    )
    parser.add_argument("--nodes", type=int, default=5000, help="Nombre de nœuds simulés")
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=300.0,
        help="Intervalle moyen d'émission en secondes",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6 * 3600.0,
        help="Durée maximale de simulation en secondes",
    )
    parser.add_argument("--seed", type=int, default=1, help="Graine de simulation")
    parser.add_argument(
        "--use-snir",
        action="store_true",
        help="Active le calcul SNIR sur les canaux",
    )
    parser.add_argument(
        "--mixra-solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Solveur utilisé pour MixRA-Opt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire de sortie pour les fichiers CSV",
    )
    parser.add_argument("--quiet", action="store_true", help="Réduit les impressions de progression")
    return parser


def _instantiate_simulator(nodes: int, packet_interval: float, seed: int, use_snir: bool) -> Simulator:
    simulator = _create_simulator(nodes, packet_interval, seed)
    simulator._interference_tracker = InterferenceTracker()
    simulator.channel.use_snir = use_snir
    multichannel = getattr(simulator, "multichannel", None)
    if multichannel is not None:
        for channel in getattr(multichannel, "channels", []):
            setattr(channel, "use_snir", use_snir)
    return simulator


def _apply_algorithm(name: str, simulator: Simulator, manager: QoSManager, solver: str) -> None:
    handler = ALGORITHMS.get(name)
    if handler is None:
        raise ValueError(f"Algorithme inconnu : {name}")

    if name == "adr":
        handler(simulator)
        return

    if name == "mixra_opt":
        handler(simulator, manager, solver)
        simulator.qos_mixra_solver = solver
        return

    handler(simulator, manager)


def main(argv: list[str] | None = None) -> Mapping[str, object]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    simulator = _instantiate_simulator(args.nodes, args.packet_interval, args.seed, args.use_snir)
    manager = QoSManager()
    _configure_clusters(manager, args.packet_interval)
    _apply_algorithm(args.algorithm, simulator, manager, args.mixra_solver)

    simulator.run(max_time=args.duration)

    metrics = simulator.get_metrics()
    metrics.update(
        {
            "num_nodes": args.nodes,
            "packet_interval_s": args.packet_interval,
            "random_seed": args.seed,
            "simulation_duration_s": getattr(simulator, "current_time", args.duration),
            "payload_bytes": PAYLOAD_BYTES,
        }
    )
    enriched = _compute_additional_metrics(simulator, dict(metrics), args.algorithm, args.mixra_solver)
    csv_row = _flatten_metrics(enriched)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.algorithm}_N{args.nodes}_T{int(args.packet_interval)}.csv"
    _write_csv(csv_path, csv_row)

    if not args.quiet:
        cluster_pdr = enriched.get("qos_cluster_pdr", {}) or {}
        print(f"Résultats enregistrés dans {csv_path}")
        if cluster_pdr:
            print("PDR par cluster : " + ", ".join(f"{k}={v:.3f}" for k, v in cluster_pdr.items()))
        print(
            f"DER={enriched.get('DER', 0.0):.3f} | Collisions={int(enriched.get('collisions', 0))} "
            f"| Jain={enriched.get('jain_index', 0.0):.3f} | Capacité={enriched.get('throughput_bps', 0.0):.1f} bps"
        )

    return {"metrics": enriched, "csv_path": csv_path}


if __name__ == "__main__":
    main()
