"""Evaluate Random Waypoint mobility under different speed and pause settings."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import MultiChannel, Simulator

from mobility_models import RandomWaypointWithPause
from simulation_analysis_utils import (
    RESULTS_DIR,
    collect_latencies,
    compute_latency_stats,
    ensure_output_directories,
    place_gateways_in_grid,
    summarise_nodes,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def derive_speed_bounds(avg_speed: float, variation: float) -> tuple[float, float]:
    min_speed = max(0.1, avg_speed * (1.0 - variation))
    max_speed = max(min_speed, avg_speed * (1.0 + variation))
    return min_speed, max_speed


def run_random_waypoint(
    avg_speed: float,
    pause: float,
    gateway_density: float,
    speed_variation: float,
    nodes: int,
    area_km: float,
    packets: int,
    interval: float,
    seed: int,
    step: float,
) -> dict:
    area_size = area_km * 1000.0
    min_speed, max_speed = derive_speed_bounds(avg_speed, speed_variation)
    rng = np.random.Generator(np.random.MT19937(seed))
    mobility = RandomWaypointWithPause(
        area_size,
        min_speed,
        max_speed,
        pause_mean=pause,
        step=step,
        rng=rng,
    )
    area_km2 = max(area_km * area_km, 1e-6)
    gateways = max(1, int(round(gateway_density * area_km2)))
    sim = Simulator(
        num_nodes=nodes,
        num_gateways=gateways,
        area_size=area_size,
        mobility=True,
        mobility_model=mobility,
        packets_to_send=packets,
        packet_interval=float(interval),
        channels=MultiChannel([868100000.0]),
        seed=seed,
    )
    place_gateways_in_grid(sim.gateways, area_size)
    sim.run()
    attempts, deliveries, energy = summarise_nodes(sim.nodes)
    latencies = collect_latencies(sim.events_log)
    stats = compute_latency_stats(latencies)
    class_pdr = sim.get_metrics().get("pdr_by_class", {})
    return {
        "avg_speed": avg_speed,
        "pause": pause,
        "gateway_density": gateway_density,
        "attempts": attempts,
        "deliveries": deliveries,
        "latencies": latencies,
        "latency_mean_s": stats["mean"],
        "latency_p95_s": stats["p95"],
        "energy_J": energy,
        "pdr_by_class": class_pdr,
    }


def aggregate(results: Iterable[dict]) -> list[dict]:
    grouped: Dict[tuple[float, float, float], dict] = {}
    for record in results:
        key = (record["avg_speed"], record["pause"], record["gateway_density"])
        bucket = grouped.setdefault(
            key,
            {
                "avg_speed": record["avg_speed"],
                "pause": record["pause"],
                "gateway_density": record["gateway_density"],
                "attempts": 0,
                "deliveries": 0,
                "latencies": [],
                "energy_J": [],
                "class_pdr": defaultdict(list),
            },
        )
        bucket["attempts"] += record["attempts"]
        bucket["deliveries"] += record["deliveries"]
        bucket["latencies"].extend(record["latencies"])
        bucket["energy_J"].extend(record["energy_J"])
        for cls, value in record["pdr_by_class"].items():
            bucket["class_pdr"][cls].append(float(value) * 100.0)
    summary: list[dict] = []
    for bucket in grouped.values():
        latencies = bucket.pop("latencies")
        energy = bucket.pop("energy_J")
        class_pdr_lists = bucket.pop("class_pdr")
        stats = compute_latency_stats(latencies)
        energy_mean = float(np.mean(energy)) if energy else 0.0
        energy_std = float(np.std(energy)) if energy else 0.0
        attempts = bucket["attempts"]
        deliveries = bucket["deliveries"]
        pdr = deliveries / attempts * 100.0 if attempts else 0.0
        class_metrics = {
            f"pdr_class_{cls}_percent": float(np.mean(values)) if values else 0.0
            for cls, values in class_pdr_lists.items()
        }
        summary.append(
            {
                **bucket,
                "pdr_percent": pdr,
                "latency_mean_s": stats["mean"],
                "latency_p95_s": stats["p95"],
                "latency_samples": len(latencies),
                "energy_mean_J": energy_mean,
                "energy_std_J": energy_std,
                **class_metrics,
            }
        )
    summary.sort(key=lambda row: (row["avg_speed"], row["pause"], row["gateway_density"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Waypoint mobility sweep")
    parser.add_argument("--avg-speeds", default="1.0,2.0", help="Comma separated average speeds (m/s)")
    parser.add_argument("--pauses", default="0,5", help="Comma separated pause means (s)")
    parser.add_argument(
        "--gateway-densities",
        default="0.05,0.1",
        help="Comma separated gateway densities (gateways per km^2)",
    )
    parser.add_argument("--speed-variation", type=float, default=0.5)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--area-km", type=float, default=10.0)
    parser.add_argument("--packets", type=int, default=50)
    parser.add_argument("--interval", type=float, default=600.0)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=float, default=1.0, help="Mobility update period (s)")
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "mobility_random_waypoint.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    avg_speeds = parse_float_list(args.avg_speeds)
    pauses = parse_float_list(args.pauses)
    densities = parse_float_list(args.gateway_densities)

    ensure_output_directories()

    replicate_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    combo_index = 0
    for avg_speed in avg_speeds:
        for pause in pauses:
            for density in densities:
                for rep in range(args.replicates):
                    seed = args.seed + rep + combo_index * 233
                    row = run_random_waypoint(
                        avg_speed,
                        pause,
                        density,
                        args.speed_variation,
                        args.nodes,
                        args.area_km,
                        args.packets,
                        args.interval,
                        seed,
                        args.step,
                    )
                    row["replicate"] = rep
                    replicate_rows.append(row)
                    aggregate_rows.append(row)
                combo_index += 1
    summary = aggregate(aggregate_rows)

    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "avg_speed",
            "pause",
            "gateway_density",
            "pdr_percent",
            "latency_mean_s",
            "latency_p95_s",
            "latency_samples",
            "attempts",
            "deliveries",
            "energy_mean_J",
            "energy_std_J",
        ]
        class_keys = sorted(k for row in summary for k in row.keys() if k.startswith("pdr_class_"))
        writer = csv.DictWriter(f, fieldnames=fieldnames + class_keys)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {args.output}")

    replicate_path = os.path.splitext(args.output)[0] + "_replicates.csv"
    with open(replicate_path, "w", newline="") as f:
        fieldnames = [
            "replicate",
            "avg_speed",
            "pause",
            "gateway_density",
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
                    "replicate": row["replicate"],
                    "avg_speed": row["avg_speed"],
                    "pause": row["pause"],
                    "gateway_density": row["gateway_density"],
                    "pdr_percent": pdr,
                    "latency_mean_s": row["latency_mean_s"],
                    "latency_p95_s": row["latency_p95_s"],
                    "attempts": attempts,
                    "deliveries": deliveries,
                    "energy_mean_J": float(np.mean(row["energy_J"])) if row["energy_J"] else 0.0,
                }
            )
    print(f"Saved replicate metrics to {replicate_path}")


if __name__ == "__main__":
    main()
