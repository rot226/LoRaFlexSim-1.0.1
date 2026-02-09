"""Utilitaires pour les scénarios SNIR statiques IWCMC (S1–S8)."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[3]
SNIR_STATIC_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SNIR_STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(SNIR_STATIC_DIR))

from loraflexsim.launcher import MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA
from loraflexsim.launcher.qos import QoSManager

from modules import aimi, apra, mixra_h, mixra_opt

FREQUENCIES_HZ = (
    868_100_000.0,
    868_300_000.0,
    868_500_000.0,
    867_100_000.0,
    867_300_000.0,
    867_500_000.0,
    867_700_000.0,
    867_900_000.0,
)

PAYLOAD_BYTES = 20


@dataclass(frozen=True)
class ScenarioConfig:
    figure_id: str
    radius_km: float
    node_counts: Sequence[int]
    packet_interval_s: float
    packets_per_node: int
    seeds: Sequence[int]
    pdr_targets: Sequence[float]
    output_path: Path


def _build_multichannel() -> MultiChannel:
    multichannel = MultiChannel(FREQUENCIES_HZ)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    for channel in multichannel.channels:
        channel.use_snir = True
        channel.advanced_capture = True
    return multichannel


def _create_simulator(
    *,
    num_nodes: int,
    area_radius_km: float,
    packet_interval_s: float,
    packets_per_node: int,
    seed: int,
) -> Simulator:
    area_size_m = area_radius_km * 2_000.0
    multichannel = _build_multichannel()
    simulator = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=area_size_m,
        transmission_mode="Random",
        packet_interval=packet_interval_s,
        first_packet_interval=packet_interval_s,
        packets_to_send=packets_per_node,
        duty_cycle=0.01,
        mobility=False,
        channels=multichannel,
        channel_distribution="round-robin",
        payload_size_bytes=PAYLOAD_BYTES,
        seed=seed,
        capture_mode="advanced",
        phy_model="omnet",
    )
    simulator.use_snir = True
    return simulator


def _apply_adr(simulator: Simulator) -> None:
    setattr(simulator, "adr_server", True)
    setattr(simulator, "adr_node", True)
    setattr(simulator, "qos_active", False)
    setattr(simulator, "qos_algorithm", "ADR")
    setattr(simulator, "qos_mixra_solver", None)


def _cluster_proportions(node_counts: Sequence[int]) -> list[float]:
    total = sum(node_counts)
    if total <= 0:
        raise ValueError("Le total de nœuds doit être strictement positif.")
    return [count / total for count in node_counts]


def _cluster_metrics(
    simulator: Simulator,
    cluster_ids: Iterable[int],
    *,
    payload_bytes: int,
) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    qos_node_clusters = getattr(simulator, "qos_node_clusters", {}) or {}
    nodes = list(getattr(simulator, "nodes", []) or [])
    events = list(getattr(simulator, "events_log", []) or [])
    sim_time = float(getattr(simulator, "current_time", 0.0) or 0.0)
    if sim_time <= 0.0:
        sim_time = 1.0
    payload_bits = float(payload_bytes) * 8.0

    for cluster_id in cluster_ids:
        node_ids = [
            node_id for node_id, cid in qos_node_clusters.items() if cid == cluster_id
        ]
        cluster_nodes = [node for node in nodes if node.id in node_ids]
        sent = sum(getattr(node, "packets_sent", 0) or 0 for node in cluster_nodes)
        attempts = sum(getattr(node, "tx_attempted", 0) or 0 for node in cluster_nodes)
        delivered = sum(getattr(node, "rx_delivered", 0) or 0 for node in cluster_nodes)

        der = delivered / sent if sent else 0.0
        pdr = delivered / attempts if attempts else 0.0
        throughput = (delivered * payload_bits / sim_time) if sim_time > 0 else 0.0

        snir_values = [
            entry.get("snir_dB")
            for entry in events
            if entry.get("node_id") in node_ids
        ]
        snir_samples = [
            float(value)
            for value in snir_values
            if value is not None and math.isfinite(float(value))
        ]
        snir_mean = float(mean(snir_samples)) if snir_samples else 0.0

        metrics[cluster_id] = {
            "cluster_nodes": int(len(cluster_nodes)),
            "der": float(der),
            "pdr": float(pdr),
            "throughput_bps": float(throughput),
            "snir_mean_db": float(snir_mean),
        }
    return metrics


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_scenario(config: ScenarioConfig) -> None:
    rows: list[dict[str, object]] = []
    proportions = _cluster_proportions(config.node_counts)
    arrival_rate = 1.0 / config.packet_interval_s if config.packet_interval_s > 0 else 0.0
    arrival_rates = [arrival_rate] * len(config.node_counts)

    algorithms = {
        "ADR": lambda sim, manager: _apply_adr(sim),
        "APRA": lambda sim, manager: apra.apply(sim, manager),
        "Aimi": lambda sim, manager: aimi.apply(sim, manager),
        "MixRA-H": lambda sim, manager: mixra_h.apply(sim, manager),
        "MixRA-Opt": lambda sim, manager: mixra_opt.apply(sim, manager),
    }

    for seed in config.seeds:
        for algo_label, algo_apply in algorithms.items():
            simulator = _create_simulator(
                num_nodes=sum(config.node_counts),
                area_radius_km=config.radius_km,
                packet_interval_s=config.packet_interval_s,
                packets_per_node=config.packets_per_node,
                seed=seed,
            )
            manager = QoSManager()
            manager.configure_clusters(
                3,
                proportions=proportions,
                arrival_rates=arrival_rates,
                pdr_targets=config.pdr_targets,
            )
            if algo_label == "ADR":
                manager._update_qos_context(simulator)
            algo_apply(simulator, manager)
            simulator.run()

            cluster_ids = [cluster.cluster_id for cluster in manager.clusters]
            cluster_metrics = _cluster_metrics(
                simulator,
                cluster_ids,
                payload_bytes=PAYLOAD_BYTES,
            )
            cluster_config = getattr(simulator, "qos_clusters_config", {}) or {}
            for cluster_id in cluster_ids:
                cluster_info = cluster_config.get(cluster_id, {})
                metrics = cluster_metrics.get(cluster_id, {})
                rows.append(
                    {
                        "figure": config.figure_id,
                        "algorithm": algo_label,
                        "seed": seed,
                        "radius_km": config.radius_km,
                        "packet_interval_s": config.packet_interval_s,
                        "packets_per_node": config.packets_per_node,
                        "cluster_id": cluster_id,
                        "cluster_nodes": metrics.get("cluster_nodes", 0.0),
                        "pdr_target": cluster_info.get("pdr_target", None),
                        "der": metrics.get("der", 0.0),
                        "throughput_bps": metrics.get("throughput_bps", 0.0),
                        "snir_mean_db": metrics.get("snir_mean_db", 0.0),
                        "pdr_achieved": metrics.get("pdr", 0.0),
                    }
                )

    _write_csv(config.output_path, rows)
    print(f"Résultats enregistrés dans {config.output_path}")
