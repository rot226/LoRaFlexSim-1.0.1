"""Balayage de densité pour UCB1 sur trois clusters QoS.

Ce script lance des simulations en faisant varier le nombre de nœuds entre
2000 et 15000. Les clusters QoS sont répartis selon des proportions fixes
(10 %, 30 %, 60 %) et chaque exécution exporte un CSV avec les champs
``["num_nodes","cluster","sf","reward_mean","reward_variance","der","pdr","snir_avg","success_rate"]``.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))

from loraflexsim.launcher import Simulator  # noqa: E402


@dataclass
class ClusterMetrics:
    """Regroupe les métriques exportées pour un cluster."""

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


def _collect_cluster_metrics(sim: Simulator, assignments: dict[int, int]) -> list[ClusterMetrics]:
    """Calcule les métriques par cluster après simulation."""

    metrics = sim.get_metrics()
    cluster_rows: list[ClusterMetrics] = []
    cluster_ids: Iterable[int] = range(1, len(CLUSTER_PROPORTIONS) + 1)
    use_snir = bool(getattr(sim, "use_snir", False))
    snir_state = "snir_on" if use_snir else "snir_off"

    for cluster_id in cluster_ids:
        nodes = [node for node in sim.nodes if assignments.get(node.id) == cluster_id]
        if not nodes:
            continue
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
        snir_values = [node.last_radio_snir for node in nodes if node.last_radio_snir is not None]
        snir_avg = sum(snir_values) / len(snir_values) if snir_values else 0.0
        der = delivered / attempts if attempts > 0 else 0.0
        pdr = float(metrics.get("qos_cluster_pdr", {}).get(cluster_id, 0.0))
        success_rate = delivered / attempts if attempts > 0 else 0.0
        cluster_rows.append(
            ClusterMetrics(
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
    use_snir: bool = True,
    output_path: Path = RESULTS_PATH,
) -> None:
    """Exécute le balayage de densité et écrit le CSV."""

    rows: list[ClusterMetrics] = []
    for index, num_nodes in enumerate(node_counts):
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
            seed=seed + index,
        )
        _apply_snir_config(sim, use_snir)
        assignments = _assign_clusters(sim)
        sim.run()
        rows.extend(_collect_cluster_metrics(sim, assignments))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_nodes",
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
