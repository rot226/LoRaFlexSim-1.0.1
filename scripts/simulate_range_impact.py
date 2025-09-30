"""Analyse the impact of coverage range, transmit power and gateway sensitivity."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import MultiChannel, Simulator

from simulation_analysis_utils import (
    RESULTS_DIR,
    collect_latencies,
    compute_latency_stats,
    ensure_output_directories,
    summarise_nodes,
)

DistanceSpec = Sequence[float]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def position_nodes_on_rings(
    sim: Simulator,
    distances_km: DistanceSpec,
    nodes_per_distance: int,
) -> Dict[int, float]:
    """Place nodes on concentric rings and return node->distance mapping."""

    mapping: Dict[int, float] = {}
    centre = sim.area_size / 2.0
    index = 0
    for dist_km in distances_km:
        radius = dist_km * 1000.0
        for j in range(nodes_per_distance):
            angle = 2.0 * math.pi * (j / nodes_per_distance)
            x = centre + radius * math.cos(angle)
            y = centre + radius * math.sin(angle)
            node = sim.nodes[index]
            node.x = node.initial_x = x
            node.y = node.initial_y = y
            mapping[node.id] = dist_km
            index += 1
    return mapping


def summarise_distance_metrics(
    sim: Simulator,
    distance_map: Dict[int, float],
) -> list[dict]:
    """Extract per-distance metrics after a simulation run."""

    events = sim.events_log
    rows: list[dict] = []
    for dist_km in sorted(set(distance_map.values())):
        node_ids = [nid for nid, value in distance_map.items() if value == dist_km]
        attempts, deliveries, energy = summarise_nodes(sim.nodes, node_ids)
        latencies = collect_latencies(events, node_ids)
        stats = compute_latency_stats(latencies)
        rows.append(
            {
                "distance_km": dist_km,
                "attempts": attempts,
                "deliveries": deliveries,
                "latencies": latencies,
                "latency_mean_s": stats["mean"],
                "latency_p95_s": stats["p95"],
                "energy_J": energy,
            }
        )
    return rows


def aggregate_results(records: Iterable[dict]) -> list[dict]:
    grouped: Dict[Tuple[float, float, float], dict] = {}
    for record in records:
        key = (
            record["distance_km"],
            record["tx_power_dBm"],
            record["gateway_threshold_dBm"],
        )
        bucket = grouped.setdefault(
            key,
            {
                "distance_km": record["distance_km"],
                "tx_power_dBm": record["tx_power_dBm"],
                "gateway_threshold_dBm": record["gateway_threshold_dBm"],
                "attempts": 0,
                "deliveries": 0,
                "latencies": [],
                "energy_J": [],
            },
        )
        bucket["attempts"] += record["attempts"]
        bucket["deliveries"] += record["deliveries"]
        bucket["latencies"].extend(record["latencies"])
        bucket["energy_J"].extend(record["energy_J"])
    summary: list[dict] = []
    for bucket in grouped.values():
        latencies = bucket.pop("latencies")
        energy = bucket.pop("energy_J")
        stats = compute_latency_stats(latencies)
        energy_mean = float(np.mean(energy)) if energy else 0.0
        energy_std = float(np.std(energy)) if energy else 0.0
        attempts = bucket["attempts"]
        deliveries = bucket["deliveries"]
        pdr = deliveries / attempts * 100.0 if attempts else 0.0
        summary.append(
            {
                **bucket,
                "pdr_percent": pdr,
                "latency_mean_s": stats["mean"],
                "latency_p95_s": stats["p95"],
                "latency_samples": len(latencies),
                "energy_mean_J": energy_mean,
                "energy_std_J": energy_std,
            }
        )
    summary.sort(key=lambda row: (row["distance_km"], row["tx_power_dBm"], row["gateway_threshold_dBm"]))
    return summary


def run_single_combination(
    distances_km: DistanceSpec,
    tx_power: float,
    threshold: float,
    nodes_per_distance: int,
    packets: int,
    interval: float,
    seed: int,
) -> list[dict]:
    max_radius_m = max(distances_km) * 1000.0
    area_size = max_radius_m * 2.2
    total_nodes = nodes_per_distance * len(distances_km)
    sim = Simulator(
        num_nodes=total_nodes,
        num_gateways=1,
        area_size=area_size,
        mobility=False,
        packets_to_send=packets,
        packet_interval=float(interval),
        channels=MultiChannel([868100000.0]),
        detection_threshold_dBm=float(threshold),
        fixed_tx_power=float(tx_power),
        seed=seed,
    )
    mapping = position_nodes_on_rings(sim, distances_km, nodes_per_distance)
    sim.run()
    rows = summarise_distance_metrics(sim, mapping)
    for row in rows:
        row.update({
            "tx_power_dBm": tx_power,
            "gateway_threshold_dBm": threshold,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate PDR/latency impact for different coverage ranges",
    )
    parser.add_argument(
        "--distances",
        default="5,10,15",
        help="Comma separated coverage distances in km",
    )
    parser.add_argument(
        "--tx-powers",
        default="14,17,20",
        help="Comma separated transmit powers in dBm",
    )
    parser.add_argument(
        "--sensitivities",
        default="-120,-125,-130",
        help="Comma separated detection thresholds in dBm",
    )
    parser.add_argument("--nodes-per-distance", type=int, default=30)
    parser.add_argument("--packets", type=int, default=50, help="Packets per node")
    parser.add_argument("--interval", type=float, default=900.0, help="Mean packet interval (s)")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "range_impact_summary.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--replicate-output",
        default=None,
        help="Optional CSV storing per replicate metrics",
    )
    args = parser.parse_args()

    distances = parse_float_list(args.distances)
    tx_powers = parse_float_list(args.tx_powers)
    sensitivities = parse_float_list(args.sensitivities)

    ensure_output_directories()

    replicate_rows: list[dict] = []
    combo_records: list[dict] = []
    combo_index = 0
    for tx_power in tx_powers:
        for threshold in sensitivities:
            for rep in range(args.replicates):
                seed = args.seed + rep + combo_index * 101
                run_rows = run_single_combination(
                    distances,
                    tx_power,
                    threshold,
                    args.nodes_per_distance,
                    args.packets,
                    args.interval,
                    seed,
                )
                for row in run_rows:
                    row_with_meta = {
                        **row,
                        "replicate": rep,
                    }
                    replicate_rows.append(row_with_meta)
                    combo_records.append(row_with_meta)
            combo_index += 1
    summary = aggregate_results(combo_records)

    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "distance_km",
            "tx_power_dBm",
            "gateway_threshold_dBm",
            "pdr_percent",
            "latency_mean_s",
            "latency_p95_s",
            "latency_samples",
            "attempts",
            "deliveries",
            "energy_mean_J",
            "energy_std_J",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {args.output}")

    if args.replicate_output:
        with open(args.replicate_output, "w", newline="") as f:
            fieldnames = [
                "replicate",
                "distance_km",
                "tx_power_dBm",
                "gateway_threshold_dBm",
                "pdr_percent",
                "latency_mean_s",
                "latency_p95_s",
                "attempts",
                "deliveries",
                "energy_mean_J",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in replicate_rows:
                attempts = row["attempts"]
                deliveries = row["deliveries"]
                pdr = deliveries / attempts * 100.0 if attempts else 0.0
                writer.writerow(
                    {
                        key: row[key]
                        for key in [
                            "replicate",
                            "distance_km",
                            "tx_power_dBm",
                            "gateway_threshold_dBm",
                        ]
                    }
                    | {
                        "pdr_percent": pdr,
                        "latency_mean_s": row["latency_mean_s"],
                        "latency_p95_s": row["latency_p95_s"],
                        "attempts": attempts,
                        "deliveries": deliveries,
                        "energy_mean_J": float(np.mean(row["energy_J"])) if row["energy_J"] else 0.0,
                    }
                )
        print(f"Saved replicate metrics to {args.replicate_output}")


if __name__ == "__main__":
    main()
