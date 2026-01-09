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
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
DEFAULT_SNIR_STATES: tuple[bool, ...] = (False, True)
SNIR_STATE_LABELS = {False: "snir_off", True: "snir_on"}


def _parse_snir_states(raw: str | None) -> Sequence[bool]:
    if raw is None:
        return DEFAULT_SNIR_STATES
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Aucun état SNIR fourni.")
    mapping = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "snir_on": True,
        "snir_off": False,
    }
    states: list[bool] = []
    for token in tokens:
        if token not in mapping:
            raise ValueError(f"État SNIR invalide: {token}")
        states.append(mapping[token])
    return tuple(states)


def _apply_snir_config(sim: Simulator, use_snir: bool) -> None:
    sim.use_snir = bool(use_snir)
    multichannel = getattr(sim, "multichannel", None)
    if multichannel is None:
        return
    for channel in multichannel.channels:
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


def _collect_cluster_metrics(
    sim: Simulator,
    assignments: dict[int, int],
    *,
    snir_state: str,
    use_snir: bool,
) -> list[ClusterMetrics]:
    """Calcule les métriques par cluster après simulation."""

    metrics = sim.get_metrics()
    cluster_rows: list[ClusterMetrics] = []
    cluster_ids: Iterable[int] = range(1, len(CLUSTER_PROPORTIONS) + 1)

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
    output_path: Path = RESULTS_PATH,
    use_snir_states: Sequence[bool] | None = None,
) -> None:
    """Exécute le balayage de densité et écrit le CSV."""

    snir_states = tuple(use_snir_states) if use_snir_states is not None else DEFAULT_SNIR_STATES
    output_dir = output_path.parent
    metrics_name = output_path.name

    for use_snir in snir_states:
        snir_state = SNIR_STATE_LABELS.get(bool(use_snir), "snir_unknown")
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
            _apply_snir_config(sim, bool(use_snir))
            assignments = _assign_clusters(sim)
            sim.run()
            rows.extend(
                _collect_cluster_metrics(
                    sim,
                    assignments,
                    snir_state=snir_state,
                    use_snir=bool(use_snir),
                )
            )

        state_dir = output_dir / snir_state
        state_dir.mkdir(parents=True, exist_ok=True)
        output_path_state = state_dir / metrics_name
        with output_path_state.open("w", newline="", encoding="utf8") as handle:
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
    parser = argparse.ArgumentParser(
        description="Balayage de densité UCB1 avec bascule SNIR."
    )
    parser.add_argument(
        "--snir-states",
        type=str,
        default=None,
        help="États SNIR séparés par des virgules (on,off,true,false,snir_on,snir_off).",
    )
    args = parser.parse_args()
    run_density_sweep(use_snir_states=_parse_snir_states(args.snir_states))
