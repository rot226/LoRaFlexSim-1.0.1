"""Balayage de charge pour UCB1 avec variation d'intervalle de paquets.

Les simulations parcourent trois intervalles fixes (5, 10 et 15 minutes)
et exportent un CSV contenant des métriques lissées par fenêtre
de longueur ``packet_interval``. Chaque ligne représente un cluster et un
indice de fenêtre, avec des moyennes pondérées par le nombre de tentatives
(``attempts``) afin de refléter la proportion de trafic réellement émise.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    reward_variance: float
    reward_window_mean: float
    reward_window_variance: float
    regret_cumulative: float
    der: float
    der_window: float
    pdr: float
    snir_avg: float
    snir_window_mean: float
    energy_avg: float
    energy_window_mean: float
    success_rate: float
    success_rate_window: float
    emission_ratio: float
    snir_state: str
    use_snir: bool


CLUSTER_PROPORTIONS = [0.1, 0.3, 0.6]
CLUSTER_TARGETS = [0.9, 0.8, 0.7]
PACKET_INTERVALS = [300.0, 600.0, 900.0]
RESULTS_PATH = ROOT / "experiments" / "ucb1" / "ucb1_load_metrics.csv"
DECISION_LOG_PATH = ROOT / "experiments" / "ucb1" / "ucb1_decision_log.csv"
DEFAULT_NODE_COUNT = 5000
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
            node.learning_method = "ucb1"
            assignments[node.id] = cluster_id
            cursor += 1
    sim.qos_clusters_config = {
        cid: {"pdr_target": target} for cid, target in enumerate(CLUSTER_TARGETS, start=1)
    }
    sim.qos_node_clusters = assignments
    return assignments


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
    assignments: dict[int, int],
    *,
    snir_state: str,
    use_snir: bool,
) -> list[ClusterMetrics]:
    metrics = sim.get_metrics()
    rows: list[ClusterMetrics] = []
    events = list(getattr(sim, "events_log", []) or [])
    packet_interval = float(getattr(sim, "packet_interval", 0.0) or 0.0)
    packets_to_send = int(getattr(sim, "packets_to_send", 0) or 0)
    node_map = {node.id: node for node in sim.nodes}

    def _bandit_reward_stats(nodes: list) -> tuple[float, float]:
        reward_means: list[float] = []
        reward_variances: list[float] = []
        for node in nodes:
            selector = getattr(node, "sf_selector", None)
            if selector and selector.bandit.total_rounds > 0:
                reward_means.extend(selector.bandit.reward_window_mean)
                reward_variances.extend(selector.bandit.reward_window_variance)
        mean = sum(reward_means) / len(reward_means) if reward_means else 0.0
        variance = (
            sum(reward_variances) / len(reward_variances) if reward_variances else 0.0
        )
        return mean, variance

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
        energy_values_cluster = [
            ev.get("energy_J")
            for ev in cluster_events
            if ev.get("energy_J") is not None and not math.isnan(float(ev.get("energy_J")))
        ]
        energy_avg = sum(energy_values_cluster) / len(energy_values_cluster) if energy_values_cluster else 0.0

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
        reward_mean_all, reward_variance_all = _bandit_reward_stats(nodes)

        pdr = float(metrics.get("qos_cluster_pdr", {}).get(cluster_id, 0.0))
        der_all = total_success / total_attempts if total_attempts > 0 else 0.0
        success_rate_all = total_success / total_attempts if total_attempts > 0 else 0.0

        window_rows: list[dict] = []
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
            energy_values = [
                ev.get("energy_J")
                for ev in window_events
                if ev.get("energy_J") is not None and not math.isnan(float(ev.get("energy_J")))
            ]
            energy_window_mean = sum(energy_values) / len(energy_values) if energy_values else 0.0

            reward_window_values = [
                _event_reward(ev, node=node_map.get(ev.get("node_id")), sim=sim) for ev in window_events
            ]
            reward_window_mean = (
                sum(reward_window_values) / len(reward_window_values)
                if reward_window_values
                else 0.0
            )
            reward_window_variance = 0.0
            if reward_window_values:
                reward_window_variance = sum(
                    (val - reward_window_mean) ** 2 for val in reward_window_values
                ) / len(reward_window_values)

            window_rows.append(
                {
                    "num_nodes": len(sim.nodes),
                    "cluster": cluster_id,
                    "packet_interval_s": packet_interval,
                    "window_index": window_index,
                    "window_start_s": window_start,
                    "window_end_s": window_end,
                    "expected_attempts": expected_attempts,
                    "attempts": attempts,
                    "sf": avg_sf,
                    "reward_mean": reward_mean_all,
                    "reward_variance": reward_variance_all,
                    "reward_window_mean": reward_window_mean,
                    "reward_window_variance": reward_window_variance,
                    "der": der_all,
                    "der_window": der_window,
                    "pdr": pdr,
                    "snir_avg": snir_avg,
                    "snir_window_mean": snir_window_mean,
                    "energy_avg": energy_avg,
                    "energy_window_mean": energy_window_mean,
                    "success_rate": success_rate_all,
                    "success_rate_window": success_rate_window,
                    "emission_ratio": emission_ratio,
                    "snir_state": snir_state,
                    "use_snir": use_snir,
                }
            )

        best_reward = max((row["reward_window_mean"] for row in window_rows), default=0.0)
        cumulative_regret = 0.0
        for row in window_rows:
            cumulative_regret += best_reward - row["reward_window_mean"]
            row["regret_cumulative"] = cumulative_regret
            rows.append(ClusterMetrics(**row))
    return rows


def _collect_decision_log(
    sim: Simulator,
    assignments: dict[int, int],
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
                "snir_state": snir_state,
                "use_snir": use_snir,
            }
        )
        decision_idx += 1
    return rows


def run_load_sweep(
    intervals: Iterable[float] = PACKET_INTERVALS,
    *,
    num_nodes: int = DEFAULT_NODE_COUNT,
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
            _apply_snir_state(sim, bool(use_snir))
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
            decision_rows.extend(
                _collect_decision_log(
                    sim,
                    assignments,
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
                    "packet_interval_s",
                    "window_index",
                    "window_start_s",
                    "window_end_s",
                    "expected_attempts",
                    "attempts",
                    "sf",
                    "reward_mean",
                    "reward_variance",
                    "reward_window_mean",
                    "reward_window_variance",
                    "regret_cumulative",
                    "der",
                    "der_window",
                    "pdr",
                    "snir_avg",
                    "snir_window_mean",
                    "energy_avg",
                    "energy_window_mean",
                    "success_rate",
                    "success_rate_window",
                    "emission_ratio",
                    "snir_state",
                    "use_snir",
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
                        f"{entry.reward_variance:.6f}",
                        f"{entry.reward_window_mean:.6f}",
                        f"{entry.reward_window_variance:.6f}",
                        f"{entry.regret_cumulative:.6f}",
                        f"{entry.der:.6f}",
                        f"{entry.der_window:.6f}",
                        f"{entry.pdr:.6f}",
                        f"{entry.snir_avg:.6f}",
                        f"{entry.snir_window_mean:.6f}",
                        f"{entry.energy_avg:.6f}",
                        f"{entry.energy_window_mean:.6f}",
                        f"{entry.success_rate:.6f}",
                        f"{entry.success_rate_window:.6f}",
                        f"{entry.emission_ratio:.6f}",
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
                            row["snir_state"],
                            str(row["use_snir"]).lower(),
                        ]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balayage de charge UCB1 avec bascule SNIR."
    )
    parser.add_argument(
        "--snir-states",
        type=str,
        default=None,
        help="États SNIR séparés par des virgules (on,off,true,false,snir_on,snir_off).",
    )
    args = parser.parse_args()
    run_load_sweep(use_snir_states=_parse_snir_states(args.snir_states))
