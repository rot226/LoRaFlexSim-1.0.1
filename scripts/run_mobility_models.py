"""Run simulations comparing different mobility models.

This utility executes scenarios for various node mobility models, collecting
PDR, collision rate, average delay and per-node energy consumption. The
aggregated metrics for each model are written to
``results/mobility_models.csv``.

Usage::

    python scripts/run_mobility_models.py --nodes 50 --packets 100 --seed 1
    python scripts/run_mobility_models.py --model random_waypoint --model smooth
    python scripts/run_mobility_models.py --model path --path-map map.json
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Callable, Dict

# Allow running the script from a clone without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loraflexsim.launcher import (
    MultiChannel,
    Simulator,
    RandomWaypoint,
    SmoothMobility,
    PathMobility,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def create_models(area_size: float, path_map_file: str | None) -> Dict[str, Callable[[], object]]:
    """Return factory functions for supported mobility models."""
    factories: Dict[str, Callable[[], object]] = {
        "random_waypoint": lambda: RandomWaypoint(area_size),
        "smooth": lambda: SmoothMobility(area_size),
    }
    if path_map_file:
        from loraflexsim.launcher.map_loader import load_map

        path_map = load_map(path_map_file)
        factories["path"] = lambda: PathMobility(area_size, path_map)
    return factories


def run_model(
    name: str,
    factory: Callable[[], object],
    num_nodes: int,
    packets: int,
    seed: int,
    adr_node: bool,
    adr_server: bool,
    area_size: float,
    interval: float,
) -> dict:
    """Run a single mobility model and return selected metrics."""
    mobility_model = factory()
    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        packets_to_send=packets,
        seed=seed,
        mobility=True,
        mobility_model=mobility_model,
        channels=MultiChannel([868100000.0]),
        adr_node=adr_node,
        adr_server=adr_server,
        area_size=area_size,
        packet_interval=interval,
    )
    sim.run()
    metrics = sim.get_metrics()
    sf_dist = metrics["sf_distribution"]
    avg_sf = sum(sf * count for sf, count in sf_dist.items()) / sum(sf_dist.values())
    total_packets = metrics["delivered"] + metrics["collisions"]
    return {
        "model": name,
        "pdr": metrics["PDR"] * 100,
        "avg_delay": metrics["avg_delay_s"],
        "energy_per_node": metrics["energy_nodes_J"] / num_nodes,
        "collision_rate": metrics["collisions"] / total_packets * 100 if total_packets else 0.0,
        "avg_sf": avg_sf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulations for multiple mobility models")
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes")
    parser.add_argument("--packets", type=int, default=100, help="Packets per node")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--adr-node", action="store_true", help="Enable ADR on nodes")
    parser.add_argument("--adr-server", action="store_true", help="Enable ADR on server")
    parser.add_argument("--area-size", type=float, default=1000.0, help="Square area size in metres")
    parser.add_argument("--interval", type=float, default=60.0, help="Mean packet interval (s)")
    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="Number of simulation replicates (>=5)",
    )
    parser.add_argument("--path-map", help="Path map file (JSON or CSV) for path mobility")
    parser.add_argument(
        "--model",
        action="append",
        choices=["random_waypoint", "smooth", "path"],
        help="Mobility model to simulate (may be repeated). Defaults to all.",
    )
    args = parser.parse_args()

    if args.replicates < 5:
        parser.error("replicates must be ≥5")

    if args.model and "path" in args.model and not args.path_map:
        parser.error("--path-map is required when using the 'path' model")

    factories = create_models(args.area_size, args.path_map)
    models = args.model or list(factories.keys())

    rows: list[dict] = []
    for model_name in models:
        factory = factories[model_name]
        rep_rows = []
        for r in range(args.replicates):
            rep_rows.append(
                run_model(
                    model_name,
                    factory,
                    args.nodes,
                    args.packets,
                    args.seed + r,
                    args.adr_node,
                    args.adr_server,
                    args.area_size,
                    args.interval,
                )
            )
        agg = {"model": model_name}
        for key in ["pdr", "collision_rate", "avg_delay", "energy_per_node", "avg_sf"]:
            values = [row[key] for row in rep_rows]
            agg[f"{key}_mean"] = statistics.mean(values)
            agg[f"{key}_std"] = statistics.pstdev(values)
        rows.append(agg)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "mobility_models.csv")
    fieldnames = [
        "model",
        "pdr_mean",
        "pdr_std",
        "collision_rate_mean",
        "collision_rate_std",
        "avg_delay_mean",
        "avg_delay_std",
        "energy_per_node_mean",
        "energy_per_node_std",
        "avg_sf_mean",
        "avg_sf_std",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
