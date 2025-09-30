"""Run a sweep to characterise class performance across node densities.

This utility executes the :class:`loraflexsim.launcher.simulator.Simulator`
for LoRaWAN classes A, B and C while varying the number of nodes participating
in the network.  The packet interval is kept constant while running several
replicates for every class/density pair.  The per-replicate metrics are written
to ``results/mne3sd/article_a/class_density_metrics.csv`` for later analysis.

Example usage::

    python scripts/mne3sd/article_a/scenarios/run_class_density_sweep.py \
        --nodes-list 50,100,250,500 --interval 300 --replicates 5 --seed 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

# Allow running the script from a clone without installation
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)

from loraflexsim.launcher import Simulator  # noqa: E402
from scripts.mne3sd.common import (
    add_execution_profile_argument,
    add_worker_argument,
    execute_simulation_tasks,
    filter_completed_tasks,
    resolve_execution_profile,
    resolve_worker_count,
    summarise_metrics,
    write_csv,
)


LOGGER = logging.getLogger("class_density_sweep")
DEFAULT_NODE_COUNTS = [50, 100, 250, 500]
FAST_NODE_COUNTS = [50, 100, 150]
FAST_REPLICATES = 3
FAST_PACKETS_PER_NODE = 20
CI_NODE_COUNTS = [20]
CI_REPLICATES = 1
CI_PACKETS_PER_NODE = 5
ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_a" / "class_density_metrics.csv"


def positive_int(value: str) -> int:
    """Return ``value`` converted to a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def parse_nodes_list(
    values: Iterable[str] | None, *, default: Iterable[int] | None = None
) -> list[int]:
    """Parse ``--nodes-list`` entries into a unique, ordered list of integers."""

    if not values:
        base = list(default) if default is not None else DEFAULT_NODE_COUNTS
        return [int(entry) for entry in base]

    nodes: list[int] = []
    seen: set[int] = set()
    for entry in values:
        for part in str(entry).split(","):
            part = part.strip()
            if not part:
                continue
            count = positive_int(part)
            if count not in seen:
                nodes.append(count)
                seen.add(count)
    if not nodes:
        raise argparse.ArgumentTypeError("--nodes-list produced an empty list")
    return nodes


def configure_logging(verbose: bool, quiet: bool) -> None:
    """Initialise logging with or without debug output."""
    if quiet:
        level = logging.WARNING
    else:
        level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def run_single_simulation(task: dict[str, object]) -> dict[str, object]:
    """Execute a single simulation and return the collected metrics."""

    class_type = str(task["class"])
    num_nodes = int(task["nodes"])
    seed = int(task["seed"])
    replicate = int(task["replicate"])

    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=int(task["gateways"]),
        packets_to_send=int(task["packets"]),
        seed=seed,
        node_class=class_type,
        packet_interval=float(task["interval"]),
        adr_node=bool(task["adr_node"]),
        adr_server=bool(task["adr_server"]),
    )
    sim.run()
    metrics = sim.get_metrics()

    delivered = int(metrics.get("delivered", 0))
    collisions = int(metrics.get("collisions", 0))
    total_packets = delivered + collisions
    collision_rate = collisions / total_packets if total_packets else 0.0
    energy_per_node = (
        metrics.get("energy_nodes_J", 0.0) / num_nodes if num_nodes else 0.0
    )
    pdr = float(metrics.get("PDR", 0.0))
    avg_delay = float(metrics.get("avg_delay_s", 0.0))

    return {
        "class": class_type,
        "nodes": num_nodes,
        "replicate": replicate,
        "pdr": pdr,
        "collision_rate": collision_rate,
        "avg_delay_s": avg_delay,
        "energy_per_node_J": energy_per_node,
    }


def main() -> None:  # noqa: D401 - CLI entry point
    parser = argparse.ArgumentParser(
        description="Run class density simulations for LoRaWAN classes A/B/C"
    )
    parser.add_argument(
        "--nodes-list",
        action="append",
        help=(
            "Comma separated list of node counts. Can be repeated. "
            "Defaults to 50,100,250,500 when omitted."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Packet interval in seconds (kept constant across runs)",
    )
    parser.add_argument(
        "--packets", type=positive_int, default=40, help="Packets to send per node"
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=5,
        help="Number of replicates for each class/node combination",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument(
        "--gateway",
        type=positive_int,
        default=1,
        help="Number of gateways for all simulations",
    )
    parser.add_argument(
        "--adr-node",
        action="store_true",
        help="Enable ADR on the nodes",
    )
    parser.add_argument(
        "--adr-server",
        action="store_true",
        help="Enable ADR on the server",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs (only warnings and the summary are printed)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip simulations that already exist in the detailed CSV",
    )
    add_worker_argument(parser, default="auto")
    add_execution_profile_argument(parser)
    args = parser.parse_args()

    configure_logging(args.verbose, args.quiet)

    profile = resolve_execution_profile(args.profile)
    if profile == "ci":
        node_defaults = CI_NODE_COUNTS
    elif profile == "fast":
        node_defaults = FAST_NODE_COUNTS
    else:
        node_defaults = DEFAULT_NODE_COUNTS
    node_counts = parse_nodes_list(args.nodes_list, default=node_defaults)
    if profile == "ci" and args.nodes_list:
        node_counts = node_counts[:1]
    if profile == "fast":
        clamped: list[int] = []
        seen: set[int] = set()
        for count in node_counts:
            limited = min(count, FAST_NODE_COUNTS[-1])
            if limited not in seen:
                clamped.append(limited)
                seen.add(limited)
        node_counts = clamped

    if profile == "ci":
        replicates = CI_REPLICATES
        packets = min(args.packets, CI_PACKETS_PER_NODE)
    elif profile == "fast":
        replicates = min(args.replicates, FAST_REPLICATES)
        packets = min(args.packets, FAST_PACKETS_PER_NODE)
    else:
        replicates = args.replicates
        packets = args.packets

    classes = ["A", "B", "C"]
    results: list[dict[str, object]] = []
    summary_rows: list[dict[str, float | str | int]] = []

    tasks: list[dict[str, object]] = []

    for class_index, class_type in enumerate(classes):
        LOGGER.info("=== Simulating class %s ===", class_type)
        for node_index, num_nodes in enumerate(node_counts):
            LOGGER.info("Node count %d", num_nodes)
            for replicate in range(1, replicates + 1):
                combination_offset = (
                    (class_index * len(node_counts) + node_index) * replicates
                )
                seed = args.seed + combination_offset + replicate - 1
                tasks.append(
                    {
                        "class": class_type,
                        "nodes": num_nodes,
                        "replicate": replicate,
                        "seed": seed,
                        "gateways": args.gateway,
                        "packets": packets,
                        "interval": args.interval,
                        "adr_node": args.adr_node,
                        "adr_server": args.adr_server,
                    }
                )

    if args.resume and RESULTS_PATH.exists():
        original_count = len(tasks)
        tasks = filter_completed_tasks(
            RESULTS_PATH, ("class", "nodes", "replicate"), tasks
        )
        skipped = original_count - len(tasks)
        LOGGER.info("Skipping %d previously completed task(s) thanks to --resume", skipped)

    worker_count = resolve_worker_count(args.workers, len(tasks))
    if worker_count > 1:
        LOGGER.info("Using %d worker processes", worker_count)

    def log_progress(task: dict[str, object], result: dict[str, object], index: int) -> None:
        LOGGER.info(
            "Class %s, nodes %d replicate %d -> PDR %.3f, collisions %.3f, avg delay %.3fs, "
            "energy/node %.4f J",
            task["class"],
            task["nodes"],
            task["replicate"],
            result["pdr"],
            result["collision_rate"],
            result["avg_delay_s"],
            result["energy_per_node_J"],
        )

    results = execute_simulation_tasks(
        tasks,
        run_single_simulation,
        max_workers=worker_count,
        progress_callback=log_progress if worker_count == 1 else None,
    )

    results.sort(key=lambda row: (row["class"], row["nodes"], row["replicate"]))

    fieldnames = [
        "class",
        "nodes",
        "replicate",
        "pdr",
        "collision_rate",
        "avg_delay_s",
        "energy_per_node_J",
    ]
    write_csv(RESULTS_PATH, fieldnames, results)

    summary_entries = summarise_metrics(
        results,
        ["class", "nodes"],
        ["pdr"],
    )
    summary_lookup = {
        (entry["class"], entry["nodes"]): entry for entry in summary_entries
    }
    for class_type in classes:
        for num_nodes in node_counts:
            entry = summary_lookup.get((class_type, num_nodes))
            if entry:
                summary_rows.append(entry)

    print(f"Results saved to {RESULTS_PATH}")
    print()
    print("Summary (PDR mean ± std):")
    header = f"{'Class':<7}{'Nodes':>10}{'PDR mean':>14}{'PDR std':>12}"
    print(header)
    print("-" * len(header))
    for entry in summary_rows:
        print(
            f"{entry['class']:<7}{entry['nodes']:>10d}{entry['pdr_mean']:>14.3f}"
            f"{entry['pdr_std']:>12.3f}"
        )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
