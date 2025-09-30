"""Compare static deployment with Random Waypoint and smooth mobility."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import MultiChannel, Simulator

from mobility_models import RandomWaypointWithPause, SmoothMobilityWithPause
from simulation_analysis_utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    collect_latencies,
    compute_latency_stats,
    ensure_output_directories,
    place_gateways_in_grid,
    summarise_nodes,
)


def derive_speed_bounds(avg_speed: float, variation: float) -> tuple[float, float]:
    min_speed = max(0.1, avg_speed * (1.0 - variation))
    max_speed = max(min_speed, avg_speed * (1.0 + variation))
    return min_speed, max_speed


def run_static(
    nodes: int,
    gateways: int,
    area_size: float,
    packets: int,
    interval: float,
    seed: int,
) -> dict:
    sim = Simulator(
        num_nodes=nodes,
        num_gateways=gateways,
        area_size=area_size,
        mobility=False,
        packets_to_send=packets,
        packet_interval=float(interval),
        channels=MultiChannel([868100000.0]),
        seed=seed,
    )
    place_gateways_in_grid(sim.gateways, area_size)
    sim.run()
    return collect_metrics(sim, "static")


def build_random_waypoint(
    area_size: float,
    avg_speed: float,
    variation: float,
    pause: float,
    seed: int,
    step: float,
):
    min_speed, max_speed = derive_speed_bounds(avg_speed, variation)
    rng = np.random.Generator(np.random.MT19937(seed))
    return RandomWaypointWithPause(
        area_size,
        min_speed,
        max_speed,
        pause_mean=pause,
        step=step,
        rng=rng,
    )


def build_smooth(
    area_size: float,
    avg_speed: float,
    variation: float,
    pause: float,
    seed: int,
    step: float,
):
    min_speed, max_speed = derive_speed_bounds(avg_speed, variation)
    rng = np.random.Generator(np.random.MT19937(seed))
    return SmoothMobilityWithPause(
        area_size,
        min_speed,
        max_speed,
        pause_mean=pause,
        step=step,
        rng=rng,
    )


def run_mobility(
    label: str,
    mobility_model,
    nodes: int,
    gateways: int,
    area_size: float,
    packets: int,
    interval: float,
    seed: int,
) -> dict:
    sim = Simulator(
        num_nodes=nodes,
        num_gateways=gateways,
        area_size=area_size,
        mobility=True,
        mobility_model=mobility_model,
        packets_to_send=packets,
        packet_interval=float(interval),
        channels=MultiChannel([868100000.0]),
        seed=seed,
    )
    place_gateways_in_grid(sim.gateways, area_size)
    sim.run()
    return collect_metrics(sim, label)


def collect_metrics(sim: Simulator, label: str) -> dict:
    attempts, deliveries, energy = summarise_nodes(sim.nodes)
    latencies = collect_latencies(sim.events_log)
    stats = compute_latency_stats(latencies)
    metrics = sim.get_metrics()
    class_metrics = {
        f"pdr_class_{cls}_percent": float(val) * 100.0 for cls, val in metrics.get("pdr_by_class", {}).items()
    }
    return {
        "scenario": label,
        "attempts": attempts,
        "deliveries": deliveries,
        "latencies": latencies,
        "latency_mean_s": stats["mean"],
        "latency_p95_s": stats["p95"],
        "energy_J": energy,
        "pdr_percent": metrics.get("PDR", 0.0) * 100.0,
        **class_metrics,
    }


def aggregate(records: Iterable[dict]) -> list[dict]:
    grouped: Dict[str, dict] = {}
    for record in records:
        bucket = grouped.setdefault(
            record["scenario"],
            {
                "scenario": record["scenario"],
                "attempts": 0,
                "deliveries": 0,
                "latencies": [],
                "energy_J": [],
                "class_pdr": {},
            },
        )
        bucket["attempts"] += record["attempts"]
        bucket["deliveries"] += record["deliveries"]
        bucket["latencies"].extend(record["latencies"])
        bucket["energy_J"].extend(record["energy_J"])
        for key, value in record.items():
            if key.startswith("pdr_class_"):
                bucket.setdefault("class_pdr", {}).setdefault(key, []).append(value)
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
            key: float(np.mean(values)) if values else 0.0 for key, values in class_pdr_lists.items()
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
    summary.sort(key=lambda row: row["scenario"])
    return summary


def plot_comparison(rows: list[dict], path: str) -> None:
    labels = [row["scenario"] for row in rows]
    pdr = [row["pdr_percent"] for row in rows]
    latency = [row["latency_p95_s"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, pdr, color="#4c72b0")
    axes[0].set_ylabel("PDR (%)")
    axes[0].set_title("Packet Delivery Ratio")
    axes[0].set_ylim(0, 100)

    axes[1].bar(labels, latency, color="#dd8452")
    axes[1].set_ylabel("95th percentile latency (s)")
    axes[1].set_title("Latency distribution")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mobility against static baseline")
    parser.add_argument("--nodes", type=int, default=150)
    parser.add_argument("--area-km", type=float, default=10.0)
    parser.add_argument("--gateway-density", type=float, default=0.08)
    parser.add_argument("--packets", type=int, default=60)
    parser.add_argument("--interval", type=float, default=600.0)
    parser.add_argument("--avg-speed", type=float, default=2.0)
    parser.add_argument("--speed-variation", type=float, default=0.4)
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "mobility_vs_static.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--figure",
        default=os.path.join(FIGURES_DIR, "mobility_vs_static.png"),
        help="Comparison figure path",
    )
    args = parser.parse_args()

    area_size = args.area_km * 1000.0
    area_km2 = max(args.area_km * args.area_km, 1e-6)
    gateways = max(1, int(round(args.gateway_density * area_km2)))

    ensure_output_directories()

    replicate_rows: list[dict] = []
    for rep in range(args.replicates):
        seed = args.seed + rep
        replicate_rows.append(
            run_static(
                args.nodes,
                gateways,
                area_size,
                args.packets,
                args.interval,
                seed,
            )
        )
        mobility_seed = args.seed + rep + 100
        rw_model = build_random_waypoint(
            area_size,
            args.avg_speed,
            args.speed_variation,
            args.pause,
            mobility_seed,
            args.step,
        )
        replicate_rows.append(
            run_mobility(
                "random_waypoint",
                rw_model,
                args.nodes,
                gateways,
                area_size,
                args.packets,
                args.interval,
                mobility_seed,
            )
        )
        smooth_model = build_smooth(
            area_size,
            args.avg_speed,
            args.speed_variation,
            args.pause,
            mobility_seed + 200,
            args.step,
        )
        replicate_rows.append(
            run_mobility(
                "smooth",
                smooth_model,
                args.nodes,
                gateways,
                area_size,
                args.packets,
                args.interval,
                mobility_seed + 200,
            )
        )
    summary = aggregate(replicate_rows)

    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "scenario",
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
    print(f"Saved comparison table to {args.output}")

    plot_comparison(summary, args.figure)
    print(f"Saved comparison figure to {args.figure}")


if __name__ == "__main__":
    main()
