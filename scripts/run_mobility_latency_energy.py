"""Run predefined mobility and multi-channel scenarios and collect metrics.

The script executes **eight** scenarios combining two node counts (50 and 200),
mobility on or off and one, three or six channels.  Each scenario records PDR,
average delay, collision rate and energy per node, aggregating the mean and
standard deviation over the requested number of replicates.  Results are stored
in ``results/mobility_latency_energy.csv``.

Usage::

    python scripts/run_mobility_latency_energy.py --packets 100 --seed 1
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
import statistics
from typing import Mapping

# Allow running the script from a clone without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import MultiChannel, Simulator  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
AGGREGATED_KEYS = [
    "pdr",
    "avg_delay",
    "energy_per_node",
    "collision_rate",
    "avg_sf",
]


def compute_average_sf(sf_distribution: Mapping[object, int | float]) -> float:
    """Return the average spreading factor from a raw distribution."""

    total_weighted = 0.0
    total_nodes = 0
    for sf, count in sf_distribution.items():
        try:
            sf_value = int(sf)
        except (TypeError, ValueError):
            # Ignore malformed keys and continue with the remaining values.
            continue
        total_weighted += sf_value * float(count)
        total_nodes += float(count)
    if total_nodes == 0:
        return 0.0
    return total_weighted / total_nodes


def run_scenario(
    name: str,
    mobility: bool,
    channels: MultiChannel,
    num_nodes: int,
    packets: int,
    seed: int,
    adr_node: bool,
    adr_server: bool,
    area_size: float,
    interval: float,
    speed: float,
) -> dict:
    """Run a single scenario and return selected metrics."""
    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        packets_to_send=packets,
        seed=seed,
        mobility=mobility,
        channels=channels,
        adr_node=adr_node,
        adr_server=adr_server,
        area_size=area_size,
        packet_interval=interval,
        mobility_speed=(speed, speed),
    )
    sim.run()
    metrics = sim.get_metrics()
    avg_sf = compute_average_sf(metrics.get("sf_distribution", {}))
    total_packets = metrics["delivered"] + metrics["collisions"]
    return {
        "scenario": name,
        "pdr": metrics["PDR"] * 100,
        "avg_delay": metrics["avg_delay_s"],
        "energy_per_node": metrics["energy_nodes_J"] / num_nodes,
        "collision_rate": metrics["collisions"] / total_packets * 100
        if total_packets
        else 0.0,
        "avg_sf": avg_sf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mobility/static and multi-channel scenarios",
    )
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes")
    parser.add_argument("--packets", type=int, default=100, help="Packets per node")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--adr-node",
        action="store_true",
        help="Enable ADR algorithm on nodes",
    )
    parser.add_argument(
        "--adr-server",
        action="store_true",
        help="Enable ADR algorithm on server",
    )
    parser.add_argument(
        "--area-size",
        type=float,
        default=1000.0,
        help="Square area size in metres",
    )
    parser.add_argument(
        "--interval", type=float, default=60.0, help="Mean packet interval (s)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Mobility speed for nodes (m/s)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 3, 6],
        default=3,
        help="Number of channels for multi-channel scenarios",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of simulation replicates",
    )
    parser.add_argument(
        "--high-traffic",
        action="store_true",
        help="Shortcut enabling congested conditions (nodes=200, interval=1s, area=500m)",
    )
    args = parser.parse_args()

    if args.high_traffic:
        if args.nodes == parser.get_default("nodes"):
            args.nodes = 200
        if args.interval == parser.get_default("interval"):
            args.interval = 1.0
        if args.area_size == parser.get_default("area_size"):
            args.area_size = 500.0

    freq_plan = [
        868100000.0,
        868300000.0,
        868500000.0,
        867100000.0,
        867300000.0,
        867500000.0,
    ]
    scenarios = {
        "n50_c1_static": {
            "mobility": False,
            "channels": 1,
            "nodes": 50,
            "speed": 0.0,
        },
        "n50_c1_mobile": {
            "mobility": True,
            "channels": 1,
            "nodes": 50,
            "speed": args.speed,
        },
        "n50_c3_mobile": {
            "mobility": True,
            "channels": 3,
            "nodes": 50,
            "speed": args.speed,
        },
        "n50_c6_static": {
            "mobility": False,
            "channels": 6,
            "nodes": 50,
            "speed": 0.0,
        },
        "n200_c1_static": {
            "mobility": False,
            "channels": 1,
            "nodes": 200,
            "speed": 0.0,
        },
        "n200_c1_mobile": {
            "mobility": True,
            "channels": 1,
            "nodes": 200,
            "speed": args.speed,
        },
        "n200_c3_mobile": {
            "mobility": True,
            "channels": 3,
            "nodes": 200,
            "speed": args.speed,
        },
        "n200_c6_static": {
            "mobility": False,
            "channels": 6,
            "nodes": 200,
            "speed": 0.0,
        },
    }

    rows: list[dict] = []
    for name, params in scenarios.items():
        nodes = params.get("nodes", args.nodes)
        speed = params.get("speed", args.speed)
        ch_count = params.get("channels", args.channels)
        channels = MultiChannel(freq_plan[: ch_count])
        rep_rows = []
        for r in range(args.replicates):
            rep_rows.append(
                run_scenario(
                    name,
                    params["mobility"],
                    channels,
                    nodes,
                    args.packets,
                    args.seed + r,
                    args.adr_node,
                    args.adr_server,
                    args.area_size,
                    args.interval,
                    speed,
                )
            )

        agg = {
            "scenario": name,
            "nodes": nodes,
            "interval": args.interval,
            "area_size": args.area_size,
            "speed": speed,
            "channels": ch_count,
        }
        for key in AGGREGATED_KEYS:
            values = [row[key] for row in rep_rows]
            agg[f"{key}_mean"] = statistics.mean(values)
            agg[f"{key}_std"] = statistics.pstdev(values)
        rows.append(agg)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "mobility_latency_energy.csv")
    fieldnames = [
        "scenario",
        "nodes",
        "interval",
        "area_size",
        "speed",
        "channels",
    ]
    for key in AGGREGATED_KEYS:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
