"""Comparaison entre UCB1 et plusieurs algorithmes de référence.

Chaque algorithme est exécuté une fois et exporte un CSV avec les colonnes
``["num_nodes","cluster","sf","reward_mean","reward_variance","der","pdr","snir_avg","success_rate","algorithm"]``.
Les algorithmes couverts sont : UCB1, ADR-MAX, ADR-AVG, MixRA-H et MixRA-Opt.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))

from loraflexsim.launcher import Simulator  # noqa: E402
from loraflexsim.launcher.qos import QoSManager  # noqa: E402


@dataclass
class ClusterMetrics:
    num_nodes: int
    cluster: int
    sf: float
    reward_mean: float
    reward_variance: float
    der: float
    pdr: float
    snir_avg: float
    success_rate: float
    algorithm: str


CLUSTER_PROPORTIONS = [0.1, 0.3, 0.6]
CLUSTER_TARGETS = [0.9, 0.8, 0.7]
RESULTS_PATH = ROOT / "experiments" / "ucb1" / "ucb1_baseline_metrics.csv"
DEFAULT_PACKET_INTERVAL = 600.0
DEFAULT_NODE_COUNT = 5000
ALGORITHMS: Sequence[str] = ["ucb1", "ADR-MAX", "ADR-AVG", "MixRA-H", "Opt"]


def _assign_clusters(sim: Simulator) -> dict[int, int]:
    assignments: dict[int, int] = {}
    total_nodes = len(sim.nodes)
    thresholds: list[int] = []
    cumulative = 0
    for proportion in CLUSTER_PROPORTIONS:
        cumulative += int(round(proportion * total_nodes))
        thresholds.append(cumulative)
    thresholds[-1] = total_nodes

    cursor = 0
    for cluster_id, threshold in enumerate(thresholds, start=1):
        while cursor < threshold and cursor < total_nodes:
            node = sim.nodes[cursor]
            node.qos_cluster_id = cluster_id
            assignments[node.id] = cluster_id
            cursor += 1
    sim.qos_clusters_config = {
        cid: {"pdr_target": target} for cid, target in enumerate(CLUSTER_TARGETS, start=1)
    }
    sim.qos_node_clusters = assignments
    return assignments


def _apply_algorithm(sim: Simulator, name: str, packet_interval: float) -> None:
    """Configure le simulateur en fonction de l'algorithme demandé."""

    if name == "ucb1":
        sim.adr_node = False
        sim.adr_server = False
        for node in sim.nodes:
            node.learning_method = "ucb1"
        return

    if name in {"ADR-MAX", "ADR-AVG"}:
        sim.adr_node = True
        sim.adr_server = True
        sim.adr_method = "max" if name == "ADR-MAX" else "avg"
        for node in sim.nodes:
            node.learning_method = None
        return

    manager = QoSManager()
    manager.configure_clusters(
        3,
        proportions=CLUSTER_PROPORTIONS,
        arrival_rates=[1.0 / packet_interval] * 3,
        pdr_targets=CLUSTER_TARGETS,
    )
    if name == "MixRA-H":
        manager.apply(sim, "MixRA-H")
        for node in sim.nodes:
            node.learning_method = None
        return
    manager.apply(sim, "MixRA-Opt")
    for node in sim.nodes:
        node.learning_method = None


def _collect_cluster_metrics(
    sim: Simulator, assignments: Mapping[int, int], algorithm: str
) -> list[ClusterMetrics]:
    metrics = sim.get_metrics()
    rows: list[ClusterMetrics] = []
    for cluster_id in range(1, len(CLUSTER_PROPORTIONS) + 1):
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
        rows.append(
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
                algorithm=algorithm,
            )
        )
    return rows


def run_baseline_comparison(
    algorithms: Sequence[str] = ALGORITHMS,
    *,
    num_nodes: int = DEFAULT_NODE_COUNT,
    packet_interval: float = DEFAULT_PACKET_INTERVAL,
    packets_per_node: int = 5,
    seed: int = 1,
    output_path: Path = RESULTS_PATH,
) -> None:
    rows: list[ClusterMetrics] = []
    for index, name in enumerate(algorithms):
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
        assignments = _assign_clusters(sim)
        _apply_algorithm(sim, name, packet_interval)
        sim.run()
        rows.extend(_collect_cluster_metrics(sim, assignments, name))

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
                "algorithm",
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
                    entry.algorithm,
                ]
            )


if __name__ == "__main__":
    run_baseline_comparison()
