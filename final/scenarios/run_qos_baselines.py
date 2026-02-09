"""Scénario QoS de référence pour LoRaFlexSim.

Exemples CLI (Windows 11, PowerShell) :
  python final/scenarios/run_qos_baselines.py --cell-radius 2500 --nodes 1000
  python final/scenarios/run_qos_baselines.py --cell-radius 2500 --nodes 1000 --config config.ini
  python final/scenarios/run_qos_baselines.py --cell-radius 2500 --nodes 1000 --runs 10 --period 300 --duration 86400

Paramètres ajustables :
  --runs (répétitions), --seed-start (graine initiale), --period (période),
  --duration (durée simulée), --output-dir (sorties CSV), --config (INI canal).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from loraflexsim.launcher import Channel, Simulator
from loraflexsim.launcher.qos import QoSManager

DEFAULT_BANDWIDTH_HZ = 125_000.0
DEFAULT_CODING_RATE_INDEX = 0  # 0 = 4/5
DEFAULT_TX_POWER_DBM = 14.0
DEFAULT_PAYLOAD_BYTES = 20
DEFAULT_PACKET_PERIOD_S = 300.0
DEFAULT_DURATION_S = 24.0 * 3600.0
DEFAULT_RUNS = 10

CLUSTER_PROPORTIONS = (0.10, 0.30, 0.60)
CLUSTER_PDR_TARGETS = (0.90, 0.80, 0.70)

ALGORITHMS = (
    ("ADR", "ADR-Pure"),
    ("APRA", "APRA-like"),
    ("Aimi", "Aimi-like"),
    ("MixRA-H", "MixRA-H"),
    ("MixRA-Opt", "MixRA-Opt"),
)


@dataclass(frozen=True)
class ClusterRunMetrics:
    der: float
    throughput_bps: float
    snir_mean_db: float | None
    snir_samples: int


@dataclass(frozen=True)
class ClusterAggregate:
    der_mean: float
    der_std: float
    throughput_mean: float
    throughput_std: float
    snir_mean: float
    snir_std: float
    snir_available: bool
    snir_samples: int


def _build_simulator(
    num_nodes: int,
    cell_radius_m: float,
    packet_period_s: float,
    seed: int,
    config_path: Path | None,
) -> Simulator:
    channel = Channel(
        bandwidth=DEFAULT_BANDWIDTH_HZ,
        coding_rate=DEFAULT_CODING_RATE_INDEX,
    )
    channel.use_snir = True
    simulator = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=cell_radius_m * 2.0,
        transmission_mode="Random",
        packet_interval=packet_period_s,
        first_packet_interval=packet_period_s,
        packets_to_send=0,
        adr_node=False,
        adr_server=False,
        duty_cycle=0.01,
        mobility=False,
        channels=[channel],
        payload_size_bytes=DEFAULT_PAYLOAD_BYTES,
        seed=seed,
        fixed_tx_power=DEFAULT_TX_POWER_DBM,
        channel_config=config_path,
    )
    simulator.use_snir = True
    return simulator


def _configure_clusters(manager: QoSManager, packet_period_s: float) -> None:
    rate = 1.0 / packet_period_s if packet_period_s > 0 else 0.0
    manager.configure_clusters(
        3,
        proportions=CLUSTER_PROPORTIONS,
        arrival_rates=[rate, rate, rate],
        pdr_targets=CLUSTER_PDR_TARGETS,
    )


def _cluster_ids(manager: QoSManager) -> Sequence[int]:
    return [cluster.cluster_id for cluster in manager.clusters]


def _snir_by_cluster(simulator: Simulator) -> Dict[int, tuple[float, int]]:
    node_cluster: Dict[int, int] = {}
    for node in getattr(simulator, "nodes", []) or []:
        node_id = getattr(node, "id", None)
        if node_id is None:
            continue
        cluster_id = getattr(node, "qos_cluster_id", None)
        if cluster_id is None:
            continue
        node_cluster[node_id] = cluster_id

    snir_accumulator: Dict[int, tuple[float, int]] = {}
    for event in getattr(simulator, "events_log", []) or []:
        node_id = event.get("node_id")
        if node_id is None or node_id not in node_cluster:
            continue
        snir_value = event.get("snir_dB")
        if snir_value is None:
            continue
        try:
            snir_float = float(snir_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(snir_float):
            continue
        cluster_id = node_cluster[node_id]
        total, count = snir_accumulator.get(cluster_id, (0.0, 0))
        snir_accumulator[cluster_id] = (total + snir_float, count + 1)
    return snir_accumulator


def _collect_cluster_metrics(
    simulator: Simulator,
    manager: QoSManager,
    metrics: Mapping[str, object],
) -> Dict[int, ClusterRunMetrics]:
    qos_pdr = metrics.get("qos_cluster_pdr", {}) or {}
    qos_throughput = metrics.get("qos_cluster_throughput_bps", {}) or {}
    snir_data = _snir_by_cluster(simulator)

    cluster_metrics: Dict[int, ClusterRunMetrics] = {}
    for cluster_id in _cluster_ids(manager):
        der = float(qos_pdr.get(cluster_id, 0.0) or 0.0)
        throughput = float(qos_throughput.get(cluster_id, 0.0) or 0.0)
        snir_total, snir_count = snir_data.get(cluster_id, (0.0, 0))
        snir_mean = snir_total / snir_count if snir_count > 0 else None
        assert der <= 1.0 + 1e-9, f"DER > 1 pour le cluster {cluster_id}: {der}"
        cluster_metrics[cluster_id] = ClusterRunMetrics(
            der=der,
            throughput_bps=throughput,
            snir_mean_db=snir_mean,
            snir_samples=snir_count,
        )
    return cluster_metrics


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def _aggregate_cluster_runs(runs: Iterable[ClusterRunMetrics]) -> ClusterAggregate:
    ders = [run.der for run in runs]
    throughputs = [run.throughput_bps for run in runs]
    snir_values = [run.snir_mean_db for run in runs if run.snir_mean_db is not None]
    snir_samples_total = sum(run.snir_samples for run in runs)

    der_mean, der_std = _mean_std(ders)
    thr_mean, thr_std = _mean_std(throughputs)
    if snir_values:
        snir_mean, snir_std = _mean_std([float(val) for val in snir_values])
        snir_available = True
    else:
        snir_mean, snir_std = 0.0, 0.0
        snir_available = False

    return ClusterAggregate(
        der_mean=der_mean,
        der_std=der_std,
        throughput_mean=thr_mean,
        throughput_std=thr_std,
        snir_mean=snir_mean,
        snir_std=snir_std,
        snir_available=snir_available,
        snir_samples=snir_samples_total,
    )


def _validate_row(row: Mapping[str, object]) -> None:
    for key, value in row.items():
        if value is None:
            raise AssertionError(f"Valeur manquante dans le CSV ({key})")
        if isinstance(value, float) and not math.isfinite(value):
            raise AssertionError(f"Valeur non finie dans le CSV ({key})")


def _write_cluster_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_qos_baselines(
    *,
    cell_radius_m: float,
    num_nodes: int,
    config_path: Path | None,
    runs: int,
    seed_start: int,
    packet_period_s: float,
    duration_s: float,
    output_dir: Path,
) -> None:
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable: {config_path}")

    results: Dict[int, Dict[str, List[ClusterRunMetrics]]] = {}

    for label, algorithm in ALGORITHMS:
        for run_idx in range(runs):
            seed = seed_start + run_idx
            simulator = _build_simulator(
                num_nodes=num_nodes,
                cell_radius_m=cell_radius_m,
                packet_period_s=packet_period_s,
                seed=seed,
                config_path=config_path,
            )
            manager = QoSManager()
            _configure_clusters(manager, packet_period_s)
            manager._update_qos_context(simulator)

            if algorithm == "ADR-Pure":
                manager.apply(simulator, algorithm)
                manager._update_qos_context(simulator)
            else:
                manager.apply(simulator, algorithm)

            simulator.run(max_time=duration_s)
            metrics = simulator.get_metrics()
            cluster_metrics = _collect_cluster_metrics(simulator, manager, metrics)
            for cluster_id, run_metrics in cluster_metrics.items():
                results.setdefault(cluster_id, {}).setdefault(label, []).append(run_metrics)

    for cluster_id, cluster_algorithms in results.items():
        rows: List[Mapping[str, object]] = []
        for label, runs_list in cluster_algorithms.items():
            aggregate = _aggregate_cluster_runs(runs_list)
            row = {
                "cluster_id": cluster_id,
                "algorithme": label,
                "der_mean": aggregate.der_mean,
                "der_std": aggregate.der_std,
                "throughput_mean_bps": aggregate.throughput_mean,
                "throughput_std_bps": aggregate.throughput_std,
                "snir_mean_db": aggregate.snir_mean,
                "snir_std_db": aggregate.snir_std,
                "snir_available": aggregate.snir_available,
                "snir_samples": aggregate.snir_samples,
                "runs": runs,
            }
            _validate_row(row)
            rows.append(row)
        csv_path = output_dir / f"cluster_{cluster_id}.csv"
        _write_cluster_csv(csv_path, rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute des scénarios QoS/ADR et exporte des CSV par cluster."
    )
    parser.add_argument(
        "--cell-radius",
        type=float,
        required=True,
        help="Rayon de cellule (mètres).",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        required=True,
        help="Nombre total de nœuds.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Fichier INI optionnel pour surcharger les paramètres radio (section [channel]).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Nombre de répétitions (graines différentes).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Graine initiale (les runs utilisent seed_start + i).",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=DEFAULT_PACKET_PERIOD_S,
        help="Période moyenne d'émission (s).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Durée simulée (s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final/data"),
        help="Dossier de sortie des CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_qos_baselines(
        cell_radius_m=args.cell_radius,
        num_nodes=args.nodes,
        config_path=args.config,
        runs=args.runs,
        seed_start=args.seed_start,
        packet_period_s=args.period,
        duration_s=args.duration,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
