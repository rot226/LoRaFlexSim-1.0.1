"""Scénario UCB1 (sélection dynamique du SF basée sur le SNIR observé).

Exemples CLI (Windows 11, PowerShell) :
  # python final/scenarios/run_ucb1.py --cell-radius 2500 --nodes 1000
  # python final/scenarios/run_ucb1.py --cell-radius 2500 --nodes 1000 --config config.ini
  # python final/scenarios/run_ucb1.py --cell-radius 2500 --nodes 1000 --runs 10 --period 300 --duration 86400

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
from loraflexsim.learning import LoRaSFSelectorUCB1

DEFAULT_BANDWIDTH_HZ = 125_000.0
DEFAULT_CODING_RATE_INDEX = 0  # 0 = 4/5
DEFAULT_TX_POWER_DBM = 14.0
DEFAULT_PAYLOAD_BYTES = 20
DEFAULT_PACKET_PERIOD_S = 300.0
DEFAULT_DURATION_S = 24.0 * 3600.0
DEFAULT_RUNS = 10
DEFAULT_INTER_SF_COUPLING = 0.1

CLUSTER_PROPORTIONS = (0.10, 0.30, 0.60)
CLUSTER_PDR_TARGETS = (0.90, 0.80, 0.70)

SF_RANGE = tuple(range(7, 13))


@dataclass(frozen=True)
class ClusterRunMetrics:
    der: float
    snir_mean_db: float | None
    snir_samples: int
    throughput_bps: float
    sf_distribution: Dict[int, float]
    reward_total: float


@dataclass(frozen=True)
class ClusterAggregate:
    der_mean: float
    der_std: float
    snir_mean: float
    snir_std: float
    snir_available: bool
    snir_samples: int
    throughput_mean: float
    throughput_std: float
    reward_mean: float
    reward_std: float
    sf_distribution_mean: Dict[int, float]
    sf_distribution_std: Dict[int, float]


@dataclass(frozen=True)
class GlobalRunMetrics:
    der: float
    snir_mean_db: float | None
    snir_samples: int
    throughput_bps: float
    sf_distribution: Dict[int, float]
    reward_total: float


@dataclass(frozen=True)
class GlobalAggregate:
    der_mean: float
    der_std: float
    snir_mean: float
    snir_std: float
    snir_available: bool
    snir_samples: int
    throughput_mean: float
    throughput_std: float
    reward_mean: float
    reward_std: float
    sf_distribution_mean: Dict[int, float]
    sf_distribution_std: Dict[int, float]


class TrackingUCB1Selector(LoRaSFSelectorUCB1):
    """Sélecteur UCB1 qui cumule la récompense observée."""

    def __init__(self) -> None:
        super().__init__(reward_mode="snir_binary")
        self.cumulative_reward = 0.0

    def update(self, *args, **kwargs) -> float:  # type: ignore[override]
        reward = super().update(*args, **kwargs)
        self.cumulative_reward += float(reward)
        return reward


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
    channel.snir_model = True
    channel.alpha_isf = DEFAULT_INTER_SF_COUPLING
    channel.orthogonal_sf = False
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


def _snir_global(simulator: Simulator) -> tuple[float, int]:
    total = 0.0
    count = 0
    for event in getattr(simulator, "events_log", []) or []:
        snir_value = event.get("snir_dB")
        if snir_value is None:
            continue
        try:
            snir_float = float(snir_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(snir_float):
            continue
        total += snir_float
        count += 1
    return total, count


def _sf_distribution_from_cluster(metrics: Mapping[str, object], cluster_id: int) -> Dict[int, float]:
    cluster_sf_channel = metrics.get("qos_cluster_sf_channel", {}) or {}
    node_counts = metrics.get("qos_cluster_node_counts", {}) or {}
    total_nodes = int(node_counts.get(cluster_id, 0) or 0)
    sf_counts: Dict[int, int] = {sf: 0 for sf in SF_RANGE}
    cluster_entry = cluster_sf_channel.get(cluster_id, {}) or {}
    for sf_value, channel_counts in cluster_entry.items():
        try:
            sf_int = int(sf_value)
        except (TypeError, ValueError):
            continue
        if sf_int not in sf_counts:
            continue
        if isinstance(channel_counts, dict):
            sf_counts[sf_int] += sum(int(value) for value in channel_counts.values())
        else:
            try:
                sf_counts[sf_int] += int(channel_counts)
            except (TypeError, ValueError):
                continue

    if total_nodes <= 0:
        return {sf: 0.0 for sf in SF_RANGE}
    return {sf: sf_counts[sf] / total_nodes for sf in SF_RANGE}


def _sf_distribution_global(metrics: Mapping[str, object], num_nodes: int) -> Dict[int, float]:
    sf_counts = metrics.get("sf_distribution", {}) or {}
    if num_nodes <= 0:
        return {sf: 0.0 for sf in SF_RANGE}
    distribution: Dict[int, float] = {}
    for sf in SF_RANGE:
        distribution[sf] = float(sf_counts.get(sf, 0) or 0) / num_nodes
    return distribution


def _cumulative_reward_by_cluster(simulator: Simulator) -> Dict[int, float]:
    rewards: Dict[int, float] = {}
    for node in getattr(simulator, "nodes", []) or []:
        cluster_id = getattr(node, "qos_cluster_id", None)
        if cluster_id is None:
            continue
        selector = getattr(node, "sf_selector", None)
        cumulative = float(getattr(selector, "cumulative_reward", 0.0) or 0.0)
        rewards[cluster_id] = rewards.get(cluster_id, 0.0) + cumulative
    return rewards


def _cumulative_reward_global(simulator: Simulator) -> float:
    total = 0.0
    for node in getattr(simulator, "nodes", []) or []:
        selector = getattr(node, "sf_selector", None)
        total += float(getattr(selector, "cumulative_reward", 0.0) or 0.0)
    return total


def _collect_cluster_metrics(
    simulator: Simulator,
    manager: QoSManager,
    metrics: Mapping[str, object],
) -> Dict[int, ClusterRunMetrics]:
    qos_pdr = metrics.get("qos_cluster_pdr", {}) or {}
    qos_throughput = metrics.get("qos_cluster_throughput_bps", {}) or {}
    snir_data = _snir_by_cluster(simulator)
    reward_data = _cumulative_reward_by_cluster(simulator)

    cluster_metrics: Dict[int, ClusterRunMetrics] = {}
    for cluster_id in _cluster_ids(manager):
        der = float(qos_pdr.get(cluster_id, 0.0) or 0.0)
        throughput_bps = float(qos_throughput.get(cluster_id, 0.0) or 0.0)
        snir_total, snir_count = snir_data.get(cluster_id, (0.0, 0))
        snir_mean = snir_total / snir_count if snir_count > 0 else None
        sf_distribution = _sf_distribution_from_cluster(metrics, cluster_id)
        reward_total = reward_data.get(cluster_id, 0.0)
        assert der <= 1.0 + 1e-9, f"DER > 1 pour le cluster {cluster_id}: {der}"
        cluster_metrics[cluster_id] = ClusterRunMetrics(
            der=der,
            snir_mean_db=snir_mean,
            snir_samples=snir_count,
            throughput_bps=throughput_bps,
            sf_distribution=sf_distribution,
            reward_total=reward_total,
        )
    return cluster_metrics


def _collect_global_metrics(
    simulator: Simulator,
    metrics: Mapping[str, object],
) -> GlobalRunMetrics:
    der = float(metrics.get("PDR", 0.0) or 0.0)
    throughput_bps = float(metrics.get("throughput_bps", 0.0) or 0.0)
    snir_total, snir_count = _snir_global(simulator)
    snir_mean = snir_total / snir_count if snir_count > 0 else None
    sf_distribution = _sf_distribution_global(metrics, simulator.num_nodes)
    reward_total = _cumulative_reward_global(simulator)
    assert der <= 1.0 + 1e-9, f"DER > 1 pour le global: {der}"
    return GlobalRunMetrics(
        der=der,
        snir_mean_db=snir_mean,
        snir_samples=snir_count,
        throughput_bps=throughput_bps,
        sf_distribution=sf_distribution,
        reward_total=reward_total,
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def _aggregate_distributions(
    runs: Iterable[Mapping[int, float]],
) -> tuple[Dict[int, float], Dict[int, float]]:
    values_by_sf: Dict[int, List[float]] = {sf: [] for sf in SF_RANGE}
    for distribution in runs:
        for sf in SF_RANGE:
            values_by_sf[sf].append(float(distribution.get(sf, 0.0) or 0.0))
    mean_dist: Dict[int, float] = {}
    std_dist: Dict[int, float] = {}
    for sf in SF_RANGE:
        mean_dist[sf], std_dist[sf] = _mean_std(values_by_sf[sf])
    return mean_dist, std_dist


def _aggregate_cluster_runs(runs: Iterable[ClusterRunMetrics]) -> ClusterAggregate:
    ders = [run.der for run in runs]
    snir_values = [run.snir_mean_db for run in runs if run.snir_mean_db is not None]
    snir_samples_total = sum(run.snir_samples for run in runs)
    throughputs = [run.throughput_bps for run in runs]
    rewards = [run.reward_total for run in runs]
    sf_mean, sf_std = _aggregate_distributions(run.sf_distribution for run in runs)

    der_mean, der_std = _mean_std(ders)
    if snir_values:
        snir_mean, snir_std = _mean_std([float(val) for val in snir_values])
        snir_available = True
    else:
        snir_mean, snir_std = 0.0, 0.0
        snir_available = False
    throughput_mean, throughput_std = _mean_std(throughputs)
    reward_mean, reward_std = _mean_std(rewards)

    return ClusterAggregate(
        der_mean=der_mean,
        der_std=der_std,
        snir_mean=snir_mean,
        snir_std=snir_std,
        snir_available=snir_available,
        snir_samples=snir_samples_total,
        throughput_mean=throughput_mean,
        throughput_std=throughput_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
        sf_distribution_mean=sf_mean,
        sf_distribution_std=sf_std,
    )


def _aggregate_global_runs(runs: Iterable[GlobalRunMetrics]) -> GlobalAggregate:
    ders = [run.der for run in runs]
    snir_values = [run.snir_mean_db for run in runs if run.snir_mean_db is not None]
    snir_samples_total = sum(run.snir_samples for run in runs)
    throughputs = [run.throughput_bps for run in runs]
    rewards = [run.reward_total for run in runs]
    sf_mean, sf_std = _aggregate_distributions(run.sf_distribution for run in runs)

    der_mean, der_std = _mean_std(ders)
    if snir_values:
        snir_mean, snir_std = _mean_std([float(val) for val in snir_values])
        snir_available = True
    else:
        snir_mean, snir_std = 0.0, 0.0
        snir_available = False
    throughput_mean, throughput_std = _mean_std(throughputs)
    reward_mean, reward_std = _mean_std(rewards)

    return GlobalAggregate(
        der_mean=der_mean,
        der_std=der_std,
        snir_mean=snir_mean,
        snir_std=snir_std,
        snir_available=snir_available,
        snir_samples=snir_samples_total,
        throughput_mean=throughput_mean,
        throughput_std=throughput_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
        sf_distribution_mean=sf_mean,
        sf_distribution_std=sf_std,
    )


def _validate_row(row: Mapping[str, object]) -> None:
    for key, value in row.items():
        if value is None:
            raise AssertionError(f"Valeur manquante dans le CSV ({key})")
        if isinstance(value, float) and not math.isfinite(value):
            raise AssertionError(f"Valeur non finie dans le CSV ({key})")


def _write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _enforce_ucb1_nodes(simulator: Simulator) -> None:
    for node in getattr(simulator, "nodes", []) or []:
        node.adr = False
        node.learning_method = "ucb1"
        node.sf_selector = TrackingUCB1Selector()


def run_ucb1(
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

    cluster_results: Dict[int, List[ClusterRunMetrics]] = {}
    global_runs: List[GlobalRunMetrics] = []

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
        _enforce_ucb1_nodes(simulator)

        simulator.run(max_time=duration_s)
        metrics = simulator.get_metrics()

        cluster_metrics = _collect_cluster_metrics(simulator, manager, metrics)
        for cluster_id, run_metrics in cluster_metrics.items():
            cluster_results.setdefault(cluster_id, []).append(run_metrics)

        global_runs.append(_collect_global_metrics(simulator, metrics))

    for cluster_id, runs_list in cluster_results.items():
        aggregate = _aggregate_cluster_runs(runs_list)
        row: Dict[str, object] = {
            "cluster_id": cluster_id,
            "der_mean": aggregate.der_mean,
            "der_std": aggregate.der_std,
            "snir_mean_db": aggregate.snir_mean,
            "snir_std_db": aggregate.snir_std,
            "snir_available": aggregate.snir_available,
            "snir_samples": aggregate.snir_samples,
            "throughput_mean_bps": aggregate.throughput_mean,
            "throughput_std_bps": aggregate.throughput_std,
            "reward_mean": aggregate.reward_mean,
            "reward_std": aggregate.reward_std,
            "runs": runs,
        }
        for sf in SF_RANGE:
            row[f"sf{sf}_mean"] = aggregate.sf_distribution_mean[sf]
            row[f"sf{sf}_std"] = aggregate.sf_distribution_std[sf]
        _validate_row(row)
        csv_path = output_dir / f"cluster_{cluster_id}.csv"
        _write_csv(csv_path, [row])

    if global_runs:
        aggregate = _aggregate_global_runs(global_runs)
        row = {
            "scope": "global",
            "der_mean": aggregate.der_mean,
            "der_std": aggregate.der_std,
            "snir_mean_db": aggregate.snir_mean,
            "snir_std_db": aggregate.snir_std,
            "snir_available": aggregate.snir_available,
            "snir_samples": aggregate.snir_samples,
            "throughput_mean_bps": aggregate.throughput_mean,
            "throughput_std_bps": aggregate.throughput_std,
            "reward_mean": aggregate.reward_mean,
            "reward_std": aggregate.reward_std,
            "runs": runs,
        }
        for sf in SF_RANGE:
            row[f"sf{sf}_mean"] = aggregate.sf_distribution_mean[sf]
            row[f"sf{sf}_std"] = aggregate.sf_distribution_std[sf]
        _validate_row(row)
        csv_path = output_dir / "global.csv"
        _write_csv(csv_path, [row])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute un scénario UCB1 basé sur le SNIR et exporte des CSV par cluster."
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
    run_ucb1(
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
