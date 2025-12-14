"""Campagne de trafic offert SNIR (stage 1).

Ce scénario simule plusieurs tailles de réseau en mode FLoRa avec trois
clusters QoS (10 %, 30 %, 60 %). Pour chaque cluster, il calcule le trafic
« offered », les collisions et le taux d'utilisation radio à partir des
sorties du simulateur, puis exporte les résultats au format CSV dans
``data/offered_traffic.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA

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

NETWORK_SIZES = (2000, 4000, 6000, 8000, 10_000, 12_000, 15_000)
CLUSTER_SHARES = (0.1, 0.3, 0.6)
PDR_TARGETS = (0.9, 0.8, 0.7)
PACKET_INTERVAL_S = 900.0
PACKETS_PER_NODE = 10
PAYLOAD_BYTES = 20
ALGORITHM_LABEL = "baseline"
OUTPUT_PATH = ROOT_DIR / "data" / "offered_traffic.csv"


def _build_multichannel() -> MultiChannel:
    multichannel = MultiChannel(FREQUENCIES_HZ)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    for channel in multichannel.channels:
        channel.use_snir = True
    return multichannel


def _cluster_counts(total_nodes: int) -> tuple[int, int, int]:
    counts = [int(total_nodes * share) for share in CLUSTER_SHARES[:-1]]
    remaining = total_nodes - sum(counts)
    counts.append(remaining)
    return tuple(counts)


def _assign_clusters(simulator: Simulator) -> dict[int, int]:
    counts = _cluster_counts(len(simulator.nodes))
    mapping: dict[int, int] = {}
    node_index = 0
    for cluster_id, cluster_size in enumerate(counts, start=1):
        for _ in range(cluster_size):
            if node_index >= len(simulator.nodes):
                break
            node = simulator.nodes[node_index]
            mapping[node.id] = cluster_id
            setattr(node, "qos_cluster_id", cluster_id)
            node_index += 1
    setattr(simulator, "qos_clusters_config", {
        cid: {"pdr_target": PDR_TARGETS[cid - 1]} for cid in range(1, 4)
    })
    setattr(simulator, "qos_node_clusters", mapping)
    return mapping


def _packet_airtime_seconds(node) -> float:
    channel = getattr(node, "channel", None)
    if channel is None:
        return 0.0
    sf_value = getattr(node, "sf", None)
    if sf_value is None:
        return 0.0
    try:
        return float(channel.airtime(int(sf_value), payload_size=PAYLOAD_BYTES))
    except Exception:
        return 0.0


def _cluster_metrics(simulator: Simulator, cluster_map: dict[int, int]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    sim_time = getattr(simulator, "current_time", 0.0)
    payload_bits = PAYLOAD_BYTES * 8.0

    channels = getattr(simulator, "multichannel", None)
    channel_count = len(getattr(channels, "channels", []) or []) if channels else 1
    if channel_count <= 0:
        channel_count = 1

    for cluster_id in sorted(set(cluster_map.values())):
        node_ids = [nid for nid, cid in cluster_map.items() if cid == cluster_id]
        nodes = [node for node in simulator.nodes if node.id in node_ids]

        sent = sum(node.packets_sent for node in nodes)
        collisions = sum(node.packets_collision for node in nodes)

        traffic_bps = (sent * payload_bits / sim_time) if sim_time > 0 else 0.0

        total_airtime = sum(_packet_airtime_seconds(node) * node.packets_sent for node in nodes)
        utilization_rate = (total_airtime / (sim_time * channel_count)) if sim_time > 0 else 0.0

        results.append(
            {
                "cluster": f"{int(CLUSTER_SHARES[cluster_id - 1] * 100)}%",
                "algorithm": ALGORITHM_LABEL,
                "traffic_bps": traffic_bps,
                "collisions": collisions,
                "utilization_rate": utilization_rate,
            }
        )

    return results


def run_campaign() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for num_nodes in NETWORK_SIZES:
        multichannel = _build_multichannel()
        simulator = Simulator(
            num_nodes=num_nodes,
            num_gateways=1,
            area_size=5000.0,
            transmission_mode="Random",
            packet_interval=PACKET_INTERVAL_S,
            first_packet_interval=PACKET_INTERVAL_S,
            packets_to_send=PACKETS_PER_NODE,
            duty_cycle=0.01,
            mobility=False,
            channels=multichannel,
            channel_distribution="round-robin",
            payload_size_bytes=PAYLOAD_BYTES,
            flora_mode=True,
            seed=1,
        )
        cluster_map = _assign_clusters(simulator)
        simulator.run()
        for metrics in _cluster_metrics(simulator, cluster_map):
            metrics["num_nodes"] = num_nodes
            rows.append(metrics)
    return rows


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "num_nodes",
                "cluster",
                "algorithm",
                "traffic_bps",
                "collisions",
                "utilization_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    dataset = run_campaign()
    write_csv(dataset, OUTPUT_PATH)
    print(f"Résultats enregistrés dans {OUTPUT_PATH}")
