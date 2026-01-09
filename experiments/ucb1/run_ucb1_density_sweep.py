"""Balayage de densité pour UCB1 sur trois clusters QoS.

Ce script lance des simulations en faisant varier le nombre de nœuds entre
2000 et 15000. Les clusters QoS sont répartis selon des proportions fixes
(10 %, 30 %, 60 %) et chaque exécution exporte un CSV avec les champs
``["num_nodes","cluster","sf","reward_mean","reward_variance","der","pdr","snir_avg","success_rate"]``.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from random import Random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))

from loraflexsim.launcher import Simulator  # noqa: E402


@dataclass
class ClusterMetrics:
    """Regroupe les métriques exportées pour un cluster."""

    seed: int
    num_nodes: int
    cluster: int
    sf: float
    reward_mean: float
    reward_variance: float
    der: float
    pdr: float
    snir_avg: float
    success_rate: float
    snir_state: str
    use_snir: bool


CLUSTER_PROPORTIONS = [0.1, 0.3, 0.6]
CLUSTER_TARGETS = [0.9, 0.8, 0.7]
DEFAULT_NODE_SWEEP = [2000, 5000, 8000, 11000, 15000]
RESULTS_PATH = ROOT / "experiments" / "ucb1" / "ucb1_density_metrics.csv"


def _apply_snir_config(sim: Simulator, use_snir: bool) -> None:
    sim.use_snir = bool(use_snir)
    if hasattr(sim, "channel") and sim.channel is not None:
        sim.channel.use_snir = bool(use_snir)
    multichannel = getattr(sim, "multichannel", None)
    for channel in getattr(multichannel, "channels", []) or []:
        channel.use_snir = bool(use_snir)


def _assign_clusters(sim: Simulator) -> dict[int, int]:
    """Assigne chaque nœud à l'un des trois clusters QoS."""

    assignments: dict[int, int] = {}
    total_nodes = len(sim.nodes)
    thresholds: list[int] = []
    cumulative = 0
    for proportion in CLUSTER_PROPORTIONS:
        cumulative += int(round(proportion * total_nodes))
        thresholds.append(cumulative)
    # Ajuster pour s'assurer que tous les nœuds sont couverts
    thresholds[-1] = total_nodes

    cursor = 0
    for cluster_id, threshold in enumerate(thresholds, start=1):
        while cursor < threshold and cursor < total_nodes:
            node = sim.nodes[cursor]
            node.qos_cluster_id = cluster_id
            node.learning_method = "ucb1"
            assignments[node.id] = cluster_id
            cursor += 1
    sim.qos_clusters_config = {
        cid: {"pdr_target": target} for cid, target in enumerate(CLUSTER_TARGETS, start=1)
    }
    sim.qos_node_clusters = assignments
    return assignments


def _snir_with_noise(snir_db: float | None, *, sim: Simulator, noise_std_db: float) -> float | None:
    if snir_db is None or noise_std_db <= 0.0:
        return snir_db
    rng = getattr(sim, "rng", None)
    if rng is None:
        return snir_db + Random().gauss(0.0, noise_std_db)
    return float(snir_db) + float(rng.normal(0.0, noise_std_db))


def _safe_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _event_reward(event: dict, *, node, sim: Simulator, snir_noise_std_db: float) -> float:
    success = event.get("result") == "Success"
    selector = getattr(node, "sf_selector", None) if node else None
    if selector is None:
        return 1.0 if success else 0.0

    snir_db = _snir_with_noise(_safe_float(event.get("snir_dB")), sim=sim, noise_std_db=snir_noise_std_db)
    airtime = float(event.get("end_time", 0.0) or 0.0) - float(event.get("start_time", 0.0) or 0.0)
    expected_der = None
    qos_config = getattr(sim, "qos_clusters_config", {}) or {}
    qos_cluster_id = getattr(node, "qos_cluster_id", None) if node else None
    if qos_cluster_id is not None:
        expected_der = qos_config.get(qos_cluster_id, {}).get("pdr_target")

    return selector.reward_from_outcome(
        success,
        snir_db=snir_db,
        snir_threshold_db=None,
        marginal_snir_margin_db=getattr(node.channel, "marginal_snir_margin_db", None) if node else None,
        airtime_s=airtime,
        energy_j=event.get("energy_J"),
        collision=event.get("result") != "Success" and event.get("heard"),
        expected_der=expected_der,
        local_der=getattr(node, "pdr", None) if node else None,
    )


def _collect_cluster_metrics(
    sim: Simulator,
    assignments: dict[int, int],
    *,
    seed: int,
    snir_noise_std_db: float,
) -> list[ClusterMetrics]:
    """Calcule les métriques par cluster après simulation."""

    metrics = sim.get_metrics()
    cluster_rows: list[ClusterMetrics] = []
    events = list(getattr(sim, "events_log", []) or [])
    node_map = {node.id: node for node in sim.nodes}
    cluster_ids: Iterable[int] = range(1, len(CLUSTER_PROPORTIONS) + 1)
    use_snir = bool(getattr(sim, "use_snir", False))
    snir_state = "snir_on" if use_snir else "snir_off"

    for cluster_id in cluster_ids:
        nodes = [node for node in sim.nodes if assignments.get(node.id) == cluster_id]
        if not nodes:
            continue
        cluster_events = [ev for ev in events if assignments.get(ev.get("node_id")) == cluster_id]
        attempts = sum(node.tx_attempted for node in nodes)
        delivered = sum(node.rx_delivered for node in nodes)
        avg_sf = sum(node.sf for node in nodes) / len(nodes)
        reward_means: list[float] = []
        reward_variances: list[float] = []
        for node in nodes:
            selector = getattr(node, "sf_selector", None)
            if selector and selector.bandit.total_rounds > 0:
                reward_means.extend(selector.bandit.reward_window_mean)
                reward_variances.extend(selector.bandit.reward_window_variance)
        reward_mean = sum(reward_means) / len(reward_means) if reward_means else 0.0
        reward_variance = (
            sum(reward_variances) / len(reward_variances) if reward_variances else 0.0
        )
        if reward_variance == 0.0 and snir_noise_std_db > 0.0:
            reward_values = [
                _event_reward(
                    ev,
                    node=node_map.get(ev.get("node_id")),
                    sim=sim,
                    snir_noise_std_db=snir_noise_std_db,
                )
                for ev in cluster_events
            ]
            if reward_values:
                reward_mean = sum(reward_values) / len(reward_values)
                reward_variance = sum((val - reward_mean) ** 2 for val in reward_values) / len(
                    reward_values
                )
        snir_values = [
            _snir_with_noise(node.last_radio_snir, sim=sim, noise_std_db=snir_noise_std_db)
            for node in nodes
            if node.last_radio_snir is not None
        ]
        snir_avg = sum(snir_values) / len(snir_values) if snir_values else 0.0
        der = delivered / attempts if attempts > 0 else 0.0
        pdr = float(metrics.get("qos_cluster_pdr", {}).get(cluster_id, 0.0))
        success_rate = delivered / attempts if attempts > 0 else 0.0
        cluster_rows.append(
            ClusterMetrics(
                seed=seed,
                num_nodes=len(sim.nodes),
                cluster=cluster_id,
                sf=avg_sf,
                reward_mean=reward_mean,
                reward_variance=reward_variance,
                der=der,
                pdr=pdr,
                snir_avg=snir_avg,
                success_rate=success_rate,
                snir_state=snir_state,
                use_snir=use_snir,
            )
        )
    return cluster_rows


def run_density_sweep(
    node_counts: Iterable[int] = DEFAULT_NODE_SWEEP,
    *,
    packet_interval: float = 600.0,
    packets_per_node: int = 5,
    seed: int = 1,
    replications: int = 10,
    use_snir: bool = True,
    snir_fading_std: float | None = 1.5,
    snir_noise_std_db: float = 0.5,
    output_path: Path = RESULTS_PATH,
) -> None:
    """Exécute le balayage de densité et écrit le CSV."""

    if replications < 10:
        raise ValueError("Le nombre de réplications doit être supérieur ou égal à 10.")
    rows: list[ClusterMetrics] = []
    node_count_list = list(node_counts)
    for replication in range(replications):
        for index, num_nodes in enumerate(node_count_list):
            run_seed = seed + replication * len(node_count_list) + index
            sim = Simulator(
                num_nodes=num_nodes,
                num_gateways=1,
                area_size=2000.0,
                transmission_mode="Random",
                packet_interval=packet_interval,
                first_packet_interval=packet_interval,
                packets_to_send=packets_per_node,
                adr_node=False,
                adr_server=False,
                seed=run_seed,
                snir_fading_std=snir_fading_std,
            )
            _apply_snir_config(sim, use_snir)
            assignments = _assign_clusters(sim)
            sim.run()
            rows.extend(
                _collect_cluster_metrics(
                    sim,
                    assignments,
                    seed=run_seed,
                    snir_noise_std_db=snir_noise_std_db,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_nodes",
                "seed",
                "cluster",
                "sf",
                "reward_mean",
                "reward_variance",
                "der",
                "pdr",
                "snir_avg",
                "success_rate",
                "snir_state",
                "use_snir",
            ]
        )
        for entry in rows:
            writer.writerow(
                [
                    entry.num_nodes,
                    entry.seed,
                    entry.cluster,
                    f"{entry.sf:.6f}",
                    f"{entry.reward_mean:.6f}",
                    f"{entry.reward_variance:.6f}",
                    f"{entry.der:.6f}",
                    f"{entry.pdr:.6f}",
                    f"{entry.snir_avg:.6f}",
                    f"{entry.success_rate:.6f}",
                    entry.snir_state,
                    entry.use_snir,
                ]
            )


if __name__ == "__main__":
    run_density_sweep()
