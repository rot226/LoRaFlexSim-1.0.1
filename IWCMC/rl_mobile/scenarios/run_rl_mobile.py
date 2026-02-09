"""Scénarios IWCMC mobilité : UCB1 vs ADR/MixRA (export CSV)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher.qos import QoSManager
from loraflexsim.launcher.simulator import InterferenceTracker, Simulator
from loraflexsim.scenarios.qos_cluster_bench import (
    AREA_SIZE_M,
    _apply_adr_pure,
    _apply_mixra_h,
    _apply_mixra_opt,
    _compute_additional_metrics,
    _configure_clusters,
    _create_simulator,
    _effective_snir_state,
    _flatten_metrics,
    _write_csv,
)

from IWCMC.rl_mobile.mobility_models import (
    RandomWaypointMobility,
    SmoothedKalmanMobility,
    available_models,
)

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "iwcmc" / "rl_mobile"
DEFAULT_NODE_COUNTS = (100, 150, 200)
DEFAULT_PACKET_INTERVALS = (300.0,)
DEFAULT_REPLICATIONS = 3
DEFAULT_DURATION_S = 3 * 3600.0
STATE_LABELS = {True: "snir_on", False: "snir_off"}


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    requires_qos: bool
    apply: Callable[[Simulator, QoSManager, str], None]


@dataclass(frozen=True)
class MobilityRunSpec:
    key: str
    label: str
    min_speed: float
    max_speed: float


ALGORITHMS: Sequence[AlgorithmSpec] = (
    AlgorithmSpec("ucb1", "UCB1", False, lambda sim, manager, solver: _apply_ucb1(sim)),
    AlgorithmSpec("adr", "ADR pur", False, lambda sim, manager, solver: _apply_adr_pure(sim)),
    AlgorithmSpec("mixra_h", "MixRA-H", True, lambda sim, manager, solver: _apply_mixra_h(sim, manager)),
    AlgorithmSpec("mixra_opt", "MixRA-Opt", True, _apply_mixra_opt),
)


def _apply_ucb1(simulator: Simulator) -> None:
    simulator.adr_node = False
    simulator.adr_server = False
    simulator.qos_active = False
    simulator.qos_algorithm = "UCB1"
    simulator.qos_mixra_solver = None
    for node in getattr(simulator, "nodes", []) or []:
        node.adr = False
        node.learning_method = "ucb1"


def _parse_int_list(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("La liste est vide.")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("La liste doit contenir des entiers.") from exc


def _parse_float_list(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("La liste est vide.")
    try:
        return [float(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("La liste doit contenir des nombres.") from exc


def _parse_speed_pairs(value: str) -> list[tuple[float, float]]:
    pairs = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise argparse.ArgumentTypeError("Les vitesses doivent être de la forme min-max.")
        left, right = chunk.split("-", 1)
        try:
            min_speed = float(left)
            max_speed = float(right)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Vitesses invalides, ex: 1-3,3-6.") from exc
        if min_speed <= 0 or max_speed <= 0 or min_speed >= max_speed:
            raise argparse.ArgumentTypeError("Les vitesses doivent être positives et min < max.")
        pairs.append((min_speed, max_speed))
    if not pairs:
        raise argparse.ArgumentTypeError("Aucune paire de vitesses fournie.")
    return pairs


def _parse_models(value: str) -> list[str]:
    available = {key for key, _ in available_models()}
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("La liste des modèles est vide.")
    invalid = [item for item in requested if item not in available]
    if invalid:
        raise argparse.ArgumentTypeError(f"Modèles inconnus: {', '.join(invalid)}")
    return requested


def _iter_combos(nodes: Sequence[int], intervals: Sequence[float]) -> Iterable[tuple[int, float]]:
    for num_nodes in nodes:
        for interval in intervals:
            yield num_nodes, interval


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        type=_parse_int_list,
        default=list(DEFAULT_NODE_COUNTS),
        help="Liste des tailles de réseau (ex: 100,150,200).",
    )
    parser.add_argument(
        "--packet-intervals",
        type=_parse_float_list,
        default=list(DEFAULT_PACKET_INTERVALS),
        help="Liste des intervalles moyens (s).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="Nombre de répétitions par configuration.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Durée maximale de simulation (s).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Graine de base.")
    parser.add_argument(
        "--use-snir",
        action="store_true",
        dest="use_snir",
        default=True,
        help="Active le calcul SNIR (par défaut).",
    )
    parser.add_argument(
        "--no-snir",
        action="store_false",
        dest="use_snir",
        help="Désactive le calcul SNIR.",
    )
    parser.add_argument(
        "--mixra-solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Solveur utilisé pour MixRA-Opt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire de sortie.",
    )
    parser.add_argument(
        "--mobility-models",
        type=_parse_models,
        default=[key for key, _ in available_models()],
        help="Modèles de mobilité (rwp,smooth).",
    )
    parser.add_argument(
        "--mobility-speeds",
        type=_parse_speed_pairs,
        default="1-3,3-6",
        help="Paires min-max des vitesses (ex: 1-3,3-6).",
    )
    parser.add_argument(
        "--skip-downlink-validation",
        action="store_true",
        help="Ignore la validation des downlinks LoRaWAN.",
    )
    return parser


def _augment_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    tx_attempted = float(metrics.get("tx_attempted", 0.0) or 0.0)
    collisions = float(metrics.get("collisions", 0.0) or 0.0)
    collision_rate = collisions / tx_attempted if tx_attempted > 0 else 0.0
    updated = dict(metrics)
    updated["collision_rate"] = collision_rate
    return updated


def _build_mobility_model(key: str, min_speed: float, max_speed: float):
    if key == "rwp":
        return RandomWaypointMobility(AREA_SIZE_M, min_speed=min_speed, max_speed=max_speed, step=1.0)
    if key == "smooth":
        return SmoothedKalmanMobility(AREA_SIZE_M, min_speed=min_speed, max_speed=max_speed, step=1.0)
    raise ValueError(f"Modèle de mobilité inconnu: {key}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    nodes = list(args.nodes)
    intervals = list(args.packet_intervals)
    replications = int(args.replications)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    speed_pairs = args.mobility_speeds

    mobility_specs = []
    available = dict(available_models())
    for key in args.mobility_models:
        spec = available.get(key)
        if spec is None:
            continue
        for min_speed, max_speed in speed_pairs:
            mobility_specs.append(MobilityRunSpec(key, spec.label, min_speed, max_speed))

    total_runs = len(nodes) * len(intervals) * replications * len(ALGORITHMS) * len(mobility_specs)
    run_index = 0

    for mobility in mobility_specs:
        for combo_index, (num_nodes, packet_interval) in enumerate(_iter_combos(nodes, intervals)):
            for rep_idx in range(replications):
                combo_seed = int(args.seed) + combo_index * replications + rep_idx
                for spec in ALGORITHMS:
                    run_index += 1
                    print(
                        f"[{run_index}/{total_runs}] {spec.label} – {mobility.label} "
                        f"{mobility.min_speed:g}-{mobility.max_speed:g} m/s N={num_nodes} "
                        f"TX={packet_interval:g}s rep={rep_idx + 1}/{replications}"
                    )
                    simulator = _create_simulator(
                        num_nodes,
                        packet_interval,
                        combo_seed,
                        use_snir=bool(args.use_snir),
                        skip_downlink_validation=bool(args.skip_downlink_validation),
                    )
                    simulator._interference_tracker = InterferenceTracker()
                    simulator.mobility_enabled = True
                    simulator.mobility_model = _build_mobility_model(
                        mobility.key, mobility.min_speed, mobility.max_speed
                    )
                    manager = QoSManager()
                    if spec.requires_qos:
                        _configure_clusters(manager, packet_interval)
                    spec.apply(simulator, manager, str(args.mixra_solver))
                    simulator.run(max_time=float(args.duration))
                    base_metrics = simulator.get_metrics()
                    effective_snir = _effective_snir_state(simulator, bool(args.use_snir))
                    base_metrics.update(
                        {
                            "num_nodes": num_nodes,
                            "packet_interval_s": packet_interval,
                            "random_seed": combo_seed,
                            "simulation_duration_s": getattr(simulator, "current_time", args.duration),
                            "use_snir": effective_snir,
                            "with_snir": effective_snir,
                            "snir_state": STATE_LABELS.get(effective_snir, "snir_unknown"),
                            "replication_index": rep_idx + 1,
                            "mobility_model": mobility.key,
                            "mobility_label": mobility.label,
                            "mobility_speed_min": mobility.min_speed,
                            "mobility_speed_max": mobility.max_speed,
                        }
                    )
                    enriched = _compute_additional_metrics(
                        simulator, dict(base_metrics), spec.label, str(args.mixra_solver)
                    )
                    enriched = _augment_metrics(enriched)
                    csv_row = _flatten_metrics(enriched)
                    speed_label = f"{mobility.min_speed:g}-{mobility.max_speed:g}".replace(".", "p")
                    filename = (
                        f"{num_nodes}_{int(packet_interval) if float(packet_interval).is_integer() else packet_interval:g}"
                        f"_rep{rep_idx + 1}.csv"
                    )
                    csv_path = (
                        output_dir
                        / mobility.key
                        / f"speed_{speed_label}"
                        / spec.key
                        / filename
                    )
                    _write_csv(csv_path, csv_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
