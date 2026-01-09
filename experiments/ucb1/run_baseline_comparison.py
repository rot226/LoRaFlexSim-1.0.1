"""Comparaison entre UCB1 et plusieurs algorithmes de référence.

Chaque algorithme est exécuté une fois et exporte un CSV avec les colonnes
``["num_nodes","cluster","sf","reward_mean","reward_variance","der","pdr","snir_avg","success_rate","algorithm"]``.
Les algorithmes couverts sont : UCB1, ADR-MAX, ADR-AVG, MixRA-H et MixRA-Opt.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
    snir_state: str
    use_snir: bool


CLUSTER_PROPORTIONS = [0.1, 0.3, 0.6]
CLUSTER_TARGETS = [0.9, 0.8, 0.7]
RESULTS_PATH = ROOT / "experiments" / "ucb1" / "ucb1_baseline_metrics.csv"
DECISION_LOG_PATH = ROOT / "experiments" / "ucb1" / "ucb1_baseline_decision_log.csv"
DEFAULT_PACKET_INTERVAL = 600.0
DEFAULT_NODE_COUNT = 5000
ALGORITHMS: Sequence[str] = ["ucb1", "ADR-MAX", "ADR-AVG", "MixRA-H", "Opt"]
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


def _apply_snir_state(sim: Simulator, use_snir: bool) -> None:
    sim.use_snir = bool(use_snir)
    multichannel = getattr(sim, "multichannel", None)
    if multichannel is None:
        return
    for channel in multichannel.channels:
        channel.use_snir = bool(use_snir)


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


def _event_reward(event: dict, *, node, sim: Simulator) -> float:
    success = event.get("result") == "Success"
    selector = getattr(node, "sf_selector", None) if node else None
    if selector is None:
        return 1.0 if success else 0.0

    snir_db = _safe_float(event.get("snir_dB"))
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
    assignments: Mapping[int, int],
    algorithm: str,
    *,
    snir_state: str,
    use_snir: bool,
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
                snir_state=snir_state,
                use_snir=use_snir,
            )
        )
    return rows


def _collect_decision_log(
    sim: Simulator,
    assignments: Mapping[int, int],
    algorithm: str,
    packet_interval: float,
    *,
    snir_state: str,
    use_snir: bool,
) -> list[dict[str, object]]:
    events = list(getattr(sim, "events_log", []) or [])
    node_map = {node.id: node for node in sim.nodes}
    payload_bits = float(getattr(sim, "payload_size_bytes", 0.0) or 0.0) * 8.0
    decision_idx = 0
    per_node_attempts: dict[int, int] = {}
    per_cluster_attempts: dict[int, int] = {}
    per_cluster_delivered: dict[int, int] = {}
    rows: list[dict[str, object]] = []

    def _event_key(ev: dict) -> tuple[float, int]:
        return (float(ev.get("start_time", 0.0) or 0.0), int(ev.get("event_id", 0) or 0))

    for event in sorted(events, key=_event_key):
        node_id = event.get("node_id")
        if node_id is None:
            continue
        cluster_id = assignments.get(node_id)
        if cluster_id is None:
            continue
        node = node_map.get(node_id)
        per_node_attempts[node_id] = per_node_attempts.get(node_id, 0) + 1
        per_cluster_attempts[cluster_id] = per_cluster_attempts.get(cluster_id, 0) + 1
        success = event.get("result") == "Success"
        if success:
            per_cluster_delivered[cluster_id] = per_cluster_delivered.get(cluster_id, 0) + 1

        airtime = float(event.get("end_time", 0.0) or 0.0) - float(event.get("start_time", 0.0) or 0.0)
        throughput = payload_bits / airtime if success and airtime > 0 else 0.0
        pdr = (
            per_cluster_delivered.get(cluster_id, 0) / per_cluster_attempts[cluster_id]
            if per_cluster_attempts.get(cluster_id)
            else 0.0
        )
        reward = _event_reward(event, node=node, sim=sim)
        policy = "ml" if getattr(node, "learning_method", None) == "ucb1" else "heuristic"

        rows.append(
            {
                "episode_idx": per_node_attempts[node_id],
                "decision_idx": decision_idx,
                "time_s": float(event.get("start_time", 0.0) or 0.0),
                "reward": reward,
                "pdr": pdr,
                "throughput": throughput,
                "snir_db": _safe_float(event.get("snir_dB")),
                "sf": int(event.get("sf") or getattr(node, "sf", 0)),
                "tx_power": float(getattr(node, "tx_power", 0.0) or 0.0),
                "policy": policy,
                "cluster": cluster_id,
                "num_nodes": len(sim.nodes),
                "packet_interval_s": packet_interval,
                "energy_j": _safe_float(event.get("energy_J")) or 0.0,
                "algorithm": algorithm,
                "snir_state": snir_state,
                "use_snir": use_snir,
            }
        )
        decision_idx += 1
    return rows


def run_baseline_comparison(
    algorithms: Sequence[str] = ALGORITHMS,
    *,
    num_nodes: int = DEFAULT_NODE_COUNT,
    packet_interval: float = DEFAULT_PACKET_INTERVAL,
    packets_per_node: int = 5,
    seed: int = 1,
    output_path: Path = RESULTS_PATH,
    decision_log_path: Path = DECISION_LOG_PATH,
    use_snir_states: Sequence[bool] | None = None,
) -> None:
    snir_states = tuple(use_snir_states) if use_snir_states is not None else DEFAULT_SNIR_STATES
    output_dir = output_path.parent
    decision_dir = decision_log_path.parent
    metrics_name = output_path.name
    decision_name = decision_log_path.name

    for use_snir in snir_states:
        snir_state = SNIR_STATE_LABELS.get(bool(use_snir), "snir_unknown")
        rows: list[ClusterMetrics] = []
        decision_rows: list[dict[str, object]] = []
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
            _apply_snir_state(sim, bool(use_snir))
            assignments = _assign_clusters(sim)
            _apply_algorithm(sim, name, packet_interval)
            sim.run()
            rows.extend(
                _collect_cluster_metrics(
                    sim,
                    assignments,
                    name,
                    snir_state=snir_state,
                    use_snir=bool(use_snir),
                )
            )
            decision_rows.extend(
                _collect_decision_log(
                    sim,
                    assignments,
                    name,
                    packet_interval,
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
                    "algorithm",
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
                        entry.algorithm,
                        entry.snir_state,
                        str(entry.use_snir).lower(),
                    ]
                )

        if decision_rows:
            decision_state_dir = decision_dir / snir_state
            decision_state_dir.mkdir(parents=True, exist_ok=True)
            decision_log_path_state = decision_state_dir / decision_name
            with decision_log_path_state.open("w", newline="", encoding="utf8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "episode_idx",
                        "decision_idx",
                        "time_s",
                        "reward",
                        "pdr",
                        "throughput",
                        "snir_db",
                        "sf",
                        "tx_power",
                        "policy",
                        "cluster",
                        "num_nodes",
                        "packet_interval_s",
                        "energy_j",
                        "algorithm",
                        "snir_state",
                        "use_snir",
                    ]
                )
                for row in decision_rows:
                    writer.writerow(
                        [
                            row["episode_idx"],
                            row["decision_idx"],
                            f"{row['time_s']:.6f}",
                            f"{row['reward']:.6f}",
                            f"{row['pdr']:.6f}",
                            f"{row['throughput']:.6f}",
                            "" if row["snir_db"] is None else f"{row['snir_db']:.6f}",
                            row["sf"],
                            f"{row['tx_power']:.3f}",
                            row["policy"],
                            row["cluster"],
                            row["num_nodes"],
                            f"{row['packet_interval_s']:.6f}",
                            f"{row['energy_j']:.6f}",
                            row["algorithm"],
                            row["snir_state"],
                            str(row["use_snir"]).lower(),
                        ]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparaison UCB1/baselines avec bascule SNIR."
    )
    parser.add_argument(
        "--snir-states",
        type=str,
        default=None,
        help="États SNIR séparés par des virgules (on,off,true,false,snir_on,snir_off).",
    )
    args = parser.parse_args()
    run_baseline_comparison(use_snir_states=_parse_snir_states(args.snir_states))
