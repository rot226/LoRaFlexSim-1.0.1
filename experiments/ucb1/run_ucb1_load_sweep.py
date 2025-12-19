"""Balayage de charge pour UCB1 avec variation d'intervalle de paquets.

Les simulations parcourent trois intervalles fixes (5, 10 et 15 minutes)
et exportent un CSV contenant des métriques lissées par fenêtre
de longueur ``packet_interval``. Chaque ligne représente un cluster et un
indice de fenêtre, avec des moyennes pondérées par le nombre de tentatives
(``attempts``) afin de refléter la proportion de trafic réellement émise.
"""
from __future__ import annotations

import csv
import math
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
    num_nodes: int
    cluster: int
    packet_interval_s: float
    window_index: int
    window_start_s: float
    window_end_s: float
    expected_attempts: int
    attempts: int
    sf: float
    reward_mean: float
    reward_window_mean: float
    der: float
    der_window: float
    pdr: float
    snir_avg: float
    snir_window_mean: float
    success_rate: float
    success_rate_window: float
    emission_ratio: float


CLUSTER_PROPORTIONS = [0.1, 0.3, 0.6]
CLUSTER_TARGETS = [0.9, 0.8, 0.7]
PACKET_INTERVALS = [300.0, 600.0, 900.0]
RESULTS_PATH = ROOT / "experiments" / "ucb1" / "ucb1_load_metrics.csv"
DEFAULT_NODE_COUNT = 5000


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
            node.learning_method = "ucb1"
            assignments[node.id] = cluster_id
            cursor += 1
    sim.qos_clusters_config = {
        cid: {"pdr_target": target} for cid, target in enumerate(CLUSTER_TARGETS, start=1)
    }
    sim.qos_node_clusters = assignments
    return assignments


def _collect_cluster_metrics(sim: Simulator, assignments: dict[int, int]) -> list[ClusterMetrics]:
    metrics = sim.get_metrics()
    rows: list[ClusterMetrics] = []
    events = list(getattr(sim, "events_log", []) or [])
    packet_interval = float(getattr(sim, "packet_interval", 0.0) or 0.0)
    packets_to_send = int(getattr(sim, "packets_to_send", 0) or 0)
    node_map = {node.id: node for node in sim.nodes}

    def _event_reward(event: dict) -> float | None:
        node = node_map.get(event.get("node_id"))
        selector = getattr(node, "sf_selector", None) if node else None
        if selector is None:
            return None

        success = event.get("result") == "Success"
        snir_db = event.get("snir_dB")
        try:
            snir_db = None if snir_db is None or math.isnan(float(snir_db)) else float(snir_db)
        except (TypeError, ValueError):
            snir_db = None
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
            airtime_s=airtime,
            energy_j=event.get("energy_J"),
            collision=event.get("result") != "Success" and event.get("heard"),
            expected_der=expected_der,
            local_der=getattr(node, "pdr", None) if node else None,
        )

    for cluster_id in range(1, len(CLUSTER_PROPORTIONS) + 1):
        nodes = [node for node in sim.nodes if assignments.get(node.id) == cluster_id]
        if not nodes:
            continue

        cluster_events = [ev for ev in events if assignments.get(ev.get("node_id")) == cluster_id]
        snir_values_cluster = [
            ev.get("snir_dB")
            for ev in cluster_events
            if ev.get("snir_dB") is not None and not math.isnan(float(ev.get("snir_dB")))
        ]
        snir_avg = sum(snir_values_cluster) / len(snir_values_cluster) if snir_values_cluster else 0.0

        total_attempts = sum(node.tx_attempted for node in nodes)
        total_success = sum(node.rx_delivered for node in nodes)
        avg_sf = sum(node.sf for node in nodes) / len(nodes)

        latest_time = max((float(ev.get("start_time", 0.0) or 0.0) for ev in cluster_events), default=0.0)
        if packets_to_send > 0:
            window_count = packets_to_send
        elif packet_interval > 0.0 and latest_time > 0.0:
            window_count = int(math.ceil((latest_time + 1e-9) / packet_interval))
        else:
            window_count = 1

        expected_attempts = len(nodes)
        reward_values = [val for val in (_event_reward(ev) for ev in cluster_events) if val is not None]
        reward_mean_all = sum(reward_values) / len(reward_values) if reward_values else 0.0

        pdr = float(metrics.get("qos_cluster_pdr", {}).get(cluster_id, 0.0))
        der_all = total_success / total_attempts if total_attempts > 0 else 0.0
        success_rate_all = total_success / total_attempts if total_attempts > 0 else 0.0

        for window_index in range(window_count):
            window_start = packet_interval * window_index
            window_end = packet_interval * (window_index + 1)
            window_events = [
                ev
                for ev in cluster_events
                if window_start <= float(ev.get("start_time", 0.0) or 0.0) < window_end
            ]

            attempts = len(window_events)
            delivered = sum(1 for ev in window_events if ev.get("result") == "Success")
            der_window = delivered / attempts if attempts > 0 else 0.0
            success_rate_window = delivered / expected_attempts if expected_attempts > 0 else 0.0
            emission_ratio = attempts / expected_attempts if expected_attempts > 0 else 0.0

            snir_values = [
                ev.get("snir_dB")
                for ev in window_events
                if ev.get("snir_dB") is not None and not math.isnan(float(ev.get("snir_dB")))
            ]
            snir_window_mean = sum(snir_values) / len(snir_values) if snir_values else 0.0

            reward_window_values = [val for val in (_event_reward(ev) for ev in window_events) if val is not None]
            reward_window_mean = (
                sum(reward_window_values) / len(reward_window_values)
                if reward_window_values
                else 0.0
            )

            rows.append(
                ClusterMetrics(
                    num_nodes=len(sim.nodes),
                    cluster=cluster_id,
                    packet_interval_s=packet_interval,
                    window_index=window_index,
                    window_start_s=window_start,
                    window_end_s=window_end,
                    expected_attempts=expected_attempts,
                    attempts=attempts,
                    sf=avg_sf,
                    reward_mean=reward_mean_all,
                    reward_window_mean=reward_window_mean,
                    der=der_all,
                    der_window=der_window,
                    pdr=pdr,
                    snir_avg=snir_avg,
                    snir_window_mean=snir_window_mean,
                    success_rate=success_rate_all,
                    success_rate_window=success_rate_window,
                    emission_ratio=emission_ratio,
                )
            )
    return rows


def run_load_sweep(
    intervals: Iterable[float] = PACKET_INTERVALS,
    *,
    num_nodes: int = DEFAULT_NODE_COUNT,
    packets_per_node: int = 5,
    seed: int = 1,
    output_path: Path = RESULTS_PATH,
) -> None:
    rows: list[ClusterMetrics] = []
    for index, packet_interval in enumerate(intervals):
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
        sim.run()
        rows.extend(_collect_cluster_metrics(sim, assignments))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_nodes",
                "cluster",
                "packet_interval_s",
                "window_index",
                "window_start_s",
                "window_end_s",
                "expected_attempts",
                "attempts",
                "sf",
                "reward_mean",
                "reward_window_mean",
                "der",
                "der_window",
                "pdr",
                "snir_avg",
                "snir_window_mean",
                "success_rate",
                "success_rate_window",
                "emission_ratio",
            ]
        )
        for entry in rows:
            writer.writerow(
                [
                    entry.num_nodes,
                    entry.cluster,
                    f"{entry.packet_interval_s:.6f}",
                    entry.window_index,
                    f"{entry.window_start_s:.6f}",
                    f"{entry.window_end_s:.6f}",
                    entry.expected_attempts,
                    entry.attempts,
                    f"{entry.sf:.6f}",
                    f"{entry.reward_mean:.6f}",
                    f"{entry.reward_window_mean:.6f}",
                    f"{entry.der:.6f}",
                    f"{entry.der_window:.6f}",
                    f"{entry.pdr:.6f}",
                    f"{entry.snir_avg:.6f}",
                    f"{entry.snir_window_mean:.6f}",
                    f"{entry.success_rate:.6f}",
                    f"{entry.success_rate_window:.6f}",
                    f"{entry.emission_ratio:.6f}",
                ]
            )


if __name__ == "__main__":
    run_load_sweep()
