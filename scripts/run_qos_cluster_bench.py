"""Script CLI pour exécuter le banc QoS par clusters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.scenarios.qos_cluster_bench import (
    DEFAULT_NODE_COUNTS,
    DEFAULT_TX_PERIODS,
    ALGORITHMS,
    run_bench,
)
from loraflexsim.scenarios.qos_cluster_presets import (
    describe_presets,
    get_preset,
    list_presets,
)


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
        "--preset",
        choices=[preset.name for preset in list_presets()],
        default=None,
        help="Sélectionne un préréglage de scénarios (quick, baseline, full)",
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=None,
        help="Remplace la liste de charges (nombre de nœuds) à explorer",
    )
    parser.add_argument(
        "--tx-periods",
        type=float,
        nargs="+",
        default=None,
        help="Remplace les périodes d'émission (secondes) à explorer",
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
        "--mode",
        choices=["benchmark", "validation"],
        default="benchmark",
        help="Active le mode validation pour reproduire les figures du papier",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les impressions de progression",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Affiche les préréglages disponibles et quitte",
    )
    return parser


def _progress(current: int, total: int, context: Dict[str, Any]) -> None:
    num_nodes = context.get("num_nodes")
    tx = context.get("packet_interval_s")
    algo = context.get("algorithm")
    print(f"[{current}/{total}] {algo} – N={num_nodes} TX={tx:.0f}s")


def _resolve_sequences(
    preset_name: str | None,
    node_counts: Sequence[int] | None,
    tx_periods: Sequence[float] | None,
    duration: float | None,
) -> tuple[Sequence[int], Sequence[float], float | None, str | None]:
    preset = get_preset(preset_name) if preset_name else None
    resolved_nodes = tuple(node_counts) if node_counts else (
        tuple(preset.node_counts) if preset else tuple(DEFAULT_NODE_COUNTS)
    )
    resolved_periods = tuple(tx_periods) if tx_periods else (
        tuple(preset.tx_periods) if preset else tuple(DEFAULT_TX_PERIODS)
    )
    resolved_duration = (
        duration
        if duration is not None
        else (preset.simulation_duration_s if preset else None)
    )
    preset_label = preset.label if preset else None
    return resolved_nodes, resolved_periods, resolved_duration, preset_label


def main(
    argv: list[str] | None = None,
    *,
    runner=run_bench,
) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_presets:
        print(describe_presets(len(ALGORITHMS)))
        return {}

    nodes, periods, duration, preset_label = _resolve_sequences(
        args.preset,
        args.node_counts,
        args.tx_periods,
        args.duration,
    )

    summary = runner(
        node_counts=nodes,
        tx_periods=periods,
        seed=args.seed,
        output_dir=args.output_dir,
        simulation_duration_s=duration,
        mixra_solver=args.mixra_solver,
        quiet=args.quiet,
        progress_callback=None if args.quiet else _progress,
        mode=args.mode,
    )
    if not args.quiet:
        if preset_label:
            print(f"Préréglage : {preset_label}")
        print(
            "Charges : "
            + ", ".join(str(value) for value in nodes)
            + " | Périodes : "
            + ", ".join(
                f"{int(value) if float(value).is_integer() else value:g} s" for value in periods
            )
        )
        if duration is not None:
            print(f"Durée max : {duration / 3600:.1f} h")
        report_path = summary.get("report_path")
        summary_path = summary.get("summary_path")
        if report_path:
            print(f"Rapport Markdown : {report_path}")
        if summary_path:
            print(f"Résumé JSON : {summary_path}")
        validation_section = summary.get("validation", {})
        validation_path = validation_section.get("normalized_metrics_path")
        if validation_path:
            print(f"Séries normalisées : {validation_path}")
    return summary


if __name__ == "__main__":
    main()
