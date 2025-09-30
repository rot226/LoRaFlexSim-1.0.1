"""Run a sweep over mobility models and speed profiles.

This scenario executes the :class:`loraflexsim.launcher.simulator.Simulator`
for the RandomWaypoint and SmoothMobility models while varying the mobility
speed bounds provided through ``--speed-profiles``.  For each combination of
``model`` and ``speed_profile`` the script runs several replicates, gathering a
selection of metrics including packet delivery ratio (PDR), mean end-to-end
latency, latency jitter (standard deviation of per-packet delays) and the mean
energy consumption per node.

Aggregated mean and standard deviation statistics are appended to
``results/mne3sd/article_b/mobility_speed_metrics.csv``.

Example usage::

    python scripts/mne3sd/article_b/scenarios/run_mobility_speed_sweep.py \
        --nodes 200 --replicates 10 --seed 42 \
        --speed-profiles "pedestrian: (0.5, 1.5)" \
        --speed-profiles "urban: (1.5, 3.5)"
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

# Allow running the script from a clone without installation
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)

from loraflexsim.launcher import RandomWaypoint, Simulator, SmoothMobility  # noqa: E402
from scripts.mne3sd.common import (
    add_execution_profile_argument,
    add_worker_argument,
    filter_completed_tasks,
    resolve_execution_profile,
    resolve_worker_count,
    summarise_metrics,
    write_csv,
)

DEFAULT_SPEED_PROFILES: list[tuple[str, tuple[float, float]]] = [
    ("pedestrian", (0.5, 1.5)),
    ("urban", (1.5, 3.5)),
    ("vehicular", (5.0, 15.0)),
]
CI_SPEED_PROFILES = [DEFAULT_SPEED_PROFILES[0]]
CI_RANGE_KM = 5.0
CI_NODES = 40
CI_PACKETS = 10
CI_REPLICATES = 1

ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_b" / "mobility_speed_metrics.csv"

LOGGER = logging.getLogger("mobility_speed_sweep")

FIELDNAMES = [
    "model",
    "speed_profile",
    "speed_min_mps",
    "speed_max_mps",
    "replicate",
    "pdr",
    "avg_delay_s",
    "jitter_s",
    "energy_per_node_J",
    "pdr_mean",
    "pdr_std",
    "avg_delay_s_mean",
    "avg_delay_s_std",
    "jitter_s_mean",
    "jitter_s_std",
    "energy_per_node_J_mean",
    "energy_per_node_J_std",
]


def positive_int(value: str) -> int:
    """Return ``value`` converted to a strictly positive integer."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def positive_float(value: str) -> float:
    """Return ``value`` converted to a strictly positive float."""

    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_speed_profiles(values: Iterable[str] | None) -> list[tuple[str, tuple[float, float]]]:
    """Parse ``--speed-profiles`` entries into an ordered list of tuples."""

    if not values:
        return DEFAULT_SPEED_PROFILES.copy()

    profiles: list[tuple[str, tuple[float, float]]] = []
    seen: set[str] = set()

    for raw in values:
        if not raw:
            continue
        name_part, sep, range_part = raw.partition(":")
        if not sep:
            raise argparse.ArgumentTypeError(
                "speed profile entries must follow the 'name: (min, max)' format"
            )
        name = name_part.strip()
        if not name:
            raise argparse.ArgumentTypeError("speed profile name cannot be empty")
        range_str = range_part.strip()
        if range_str.startswith("(") and range_str.endswith(")"):
            range_str = range_str[1:-1]
        tokens = [token for token in range_str.replace(",", " ").split() if token]
        if len(tokens) != 2:
            raise argparse.ArgumentTypeError(
                "speed profile bounds must contain exactly two numeric values"
            )
        speed_min = positive_float(tokens[0])
        speed_max = positive_float(tokens[1])
        if speed_min > speed_max:
            raise argparse.ArgumentTypeError("minimum speed cannot exceed maximum speed")
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        profiles.append((name, (speed_min, speed_max)))

    if not profiles:
        raise argparse.ArgumentTypeError("--speed-profiles produced an empty list")

    return profiles


def compute_latency_jitter(sim: Simulator) -> float:
    """Return the standard deviation of successful packet delays."""

    delays = [
        float(entry["end_time"]) - float(entry["start_time"])
        for entry in sim.events_log
        if entry.get("result") == "Success"
    ]
    if len(delays) > 1:
        return statistics.pstdev(delays)
    return 0.0


def _run_speed_replicate(task: dict[str, object]) -> dict[str, float | str]:
    """Execute a single mobility speed replicate and return its metrics."""

    model_name = str(task["model"])
    model_factory = task["model_factory"]
    profile_name = str(task["speed_profile"])
    speed_min = float(task["speed_min_mps"])
    speed_max = float(task["speed_max_mps"])
    area_size = float(task["area_size"])
    replicate = int(task["replicate"])
    seed = int(task["seed"])
    nodes = int(task["nodes"])
    packets = int(task["packets"])
    interval = float(task["interval"])
    adr_node = bool(task["adr_node"])
    adr_server = bool(task["adr_server"])

    mobility_model = model_factory(
        area_size,
        min_speed=speed_min,
        max_speed=speed_max,
    )
    sim = Simulator(
        num_nodes=nodes,
        num_gateways=1,
        packets_to_send=packets,
        seed=seed,
        mobility=True,
        mobility_model=mobility_model,
        area_size=area_size,
        mobility_speed=(speed_min, speed_max),
        packet_interval=interval,
        adr_node=adr_node,
        adr_server=adr_server,
    )
    sim.run()
    metrics = sim.get_metrics()

    energy_nodes = float(metrics.get("energy_nodes_J", 0.0))
    energy_per_node = energy_nodes / nodes if nodes else 0.0

    return {
        "model": model_name,
        "speed_profile": profile_name,
        "speed_min_mps": speed_min,
        "speed_max_mps": speed_max,
        "replicate": replicate,
        "pdr": float(metrics.get("PDR", 0.0)),
        "avg_delay_s": float(metrics.get("avg_delay_s", 0.0)),
        "jitter_s": compute_latency_jitter(sim),
        "energy_per_node_J": energy_per_node,
    }


def main() -> None:  # noqa: D401 - CLI entry point
    parser = argparse.ArgumentParser(
        description=(
            "Run a mobility speed sweep for RandomWaypoint and SmoothMobility models"
        ),
    )
    parser.add_argument(
        "--speed-profiles",
        action="append",
        help=(
            "Speed profile definitions in the form 'name: (min, max)'. Can be repeated. "
            "Defaults to pedestrian, urban and vehicular profiles when omitted."
        ),
    )
    parser.add_argument(
        "--range-km",
        type=positive_float,
        default=10.0,
        help="Communication range expressed in kilometres used to derive the area size",
    )
    parser.add_argument("--nodes", type=positive_int, default=100, help="Number of end devices")
    parser.add_argument(
        "--packets",
        type=positive_int,
        default=50,
        help="Number of packets each node should transmit",
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=5,
        help="Number of simulation replicates for each configuration",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=300.0,
        help="Mean packet interval in seconds",
    )
    parser.add_argument("--adr-node", action="store_true", help="Enable ADR on the devices")
    parser.add_argument("--adr-server", action="store_true", help="Enable ADR on the server")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip simulations that already exist in the detailed CSV",
    )
    add_worker_argument(parser, default="auto")
    add_execution_profile_argument(parser)
    args = parser.parse_args()

    profile = resolve_execution_profile(args.profile)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    speed_profiles = parse_speed_profiles(args.speed_profiles)
    if profile == "ci":
        if args.speed_profiles:
            speed_profiles = speed_profiles[:1]
        else:
            speed_profiles = CI_SPEED_PROFILES.copy()

    range_km = args.range_km if profile != "ci" else min(args.range_km, CI_RANGE_KM)
    area_size = range_km * 2000.0

    nodes = args.nodes if profile != "ci" else min(args.nodes, CI_NODES)
    packets = args.packets if profile != "ci" else min(args.packets, CI_PACKETS)
    replicates = args.replicates if profile != "ci" else CI_REPLICATES
    workers = resolve_worker_count(args.workers, replicates)

    models = [
        ("random_waypoint", RandomWaypoint),
        ("smooth", SmoothMobility),
    ]

    results: list[dict[str, float | str]] = []
    combination_index = 0

    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)

    try:
        for model_name, model_factory in models:
            for profile_name, (speed_min, speed_max) in speed_profiles:
                base_seed = args.seed + combination_index * replicates
                tasks = [
                    {
                        "model": model_name,
                        "model_factory": model_factory,
                        "speed_profile": profile_name,
                        "speed_min_mps": speed_min,
                        "speed_max_mps": speed_max,
                        "area_size": area_size,
                        "replicate": replicate,
                        "seed": base_seed + replicate - 1,
                        "nodes": nodes,
                        "packets": packets,
                        "interval": args.interval,
                        "adr_node": args.adr_node,
                        "adr_server": args.adr_server,
                    }
                    for replicate in range(1, replicates + 1)
                ]

                if args.resume and RESULTS_PATH.exists():
                    original_count = len(tasks)
                    tasks = filter_completed_tasks(
                        RESULTS_PATH, ("model", "speed_profile", "replicate"), tasks
                    )
                    skipped = original_count - len(tasks)
                    LOGGER.info(
                        "Skipping %d previously completed task(s) thanks to --resume",
                        skipped,
                    )

                if not tasks:
                    combination_index += 1
                    continue

                if executor is not None:
                    replicate_rows = list(executor.map(_run_speed_replicate, tasks, chunksize=1))
                else:
                    replicate_rows = [_run_speed_replicate(task) for task in tasks]

                replicate_rows.sort(key=lambda row: int(row["replicate"]))

                combination_index += 1

                results.extend(replicate_rows)

                summary_entries = summarise_metrics(
                    replicate_rows,
                    ["model", "speed_profile", "speed_min_mps", "speed_max_mps"],
                    ["pdr", "avg_delay_s", "jitter_s", "energy_per_node_J"],
                )
                for entry in summary_entries:
                    aggregate_row: dict[str, float | str] = {
                        "model": entry["model"],
                        "speed_profile": entry["speed_profile"],
                        "speed_min_mps": entry["speed_min_mps"],
                        "speed_max_mps": entry["speed_max_mps"],
                        "replicate": "aggregate",
                    }
                    for metric in ("pdr", "avg_delay_s", "jitter_s", "energy_per_node_J"):
                        aggregate_row[f"{metric}_mean"] = entry.get(f"{metric}_mean", "")
                        aggregate_row[f"{metric}_std"] = entry.get(f"{metric}_std", "")
                    results.append(aggregate_row)
    finally:
        if executor is not None:
            executor.shutdown()

    write_csv(RESULTS_PATH, FIELDNAMES, results)

    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
