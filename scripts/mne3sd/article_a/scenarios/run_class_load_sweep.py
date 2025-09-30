"""Run a sweep to characterise LoRaWAN class load performance.

This script launches the :class:`loraflexsim.launcher.simulator.Simulator` for
classes A, B and C across multiple traffic intervals.  For each replicate the
packet delivery ratio (PDR), collisions and energy statistics are stored in a
CSV file for subsequent analysis.

Example usage::

    python scripts/mne3sd/article_a/scenarios/run_class_load_sweep.py \
        --nodes 50 --packets 40 --replicates 10 --seed 3 --gateway 2
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
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
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


LOGGER = logging.getLogger("class_load_sweep")
DEFAULT_INTERVALS = [60.0, 300.0, 900.0]
CI_INTERVALS = [300.0]
CI_REPLICATES = 1
CI_NODES = 20
CI_PACKETS = 10
ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "results" / "mne3sd" / "article_a" / "class_load_metrics.csv"


def positive_int(value: str) -> int:
    """Return ``value`` converted to a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def parse_intervals(interval: float | None, interval_list: Iterable[float] | None) -> list[float]:
    """Combine the different interval arguments into a sorted list."""
    values: list[float] = []
    if interval_list:
        values.extend(float(v) for v in interval_list)
    if interval is not None:
        values.append(float(interval))
    if not values:
        values = DEFAULT_INTERVALS.copy()
    # Remove duplicates while preserving order
    seen: set[float] = set()
    ordered: list[float] = []
    for val in values:
        if val not in seen:
            ordered.append(val)
            seen.add(val)
    return ordered


def configure_logging(verbose: bool) -> None:
    """Initialise logging with or without debug output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def compute_energy_breakdown(metrics: dict, num_nodes: int) -> tuple[float, float, float]:
    """Return average per-node energy spent in TX, RX and sleep states."""
    breakdown = metrics.get("energy_breakdown_by_node", {})
    total_tx = 0.0
    total_rx = 0.0
    total_sleep = 0.0
    for node_breakdown in breakdown.values():
        total_tx += float(node_breakdown.get("tx", 0.0))
        rx_energy = float(node_breakdown.get("rx", 0.0))
        rx_energy += float(node_breakdown.get("listen", 0.0))
        total_rx += rx_energy
        total_sleep += float(node_breakdown.get("sleep", 0.0))
    if num_nodes <= 0:
        return 0.0, 0.0, 0.0
    return total_tx / num_nodes, total_rx / num_nodes, total_sleep / num_nodes


def run_single_simulation(task: dict[str, object]) -> dict[str, object]:
    """Execute one class/interval replicate and return the collected metrics."""

    sim = Simulator(
        num_nodes=int(task["nodes"]),
        num_gateways=int(task["gateways"]),
        packets_to_send=int(task["packets"]),
        seed=int(task["seed"]),
        node_class=str(task["class"]),
        packet_interval=float(task["interval_s"]),
        adr_node=bool(task["adr_node"]),
        adr_server=bool(task["adr_server"]),
    )
    sim.run()
    metrics = sim.get_metrics()

    num_nodes = int(task["nodes"])
    energy_per_node = metrics.get("energy_nodes_J", 0.0) / num_nodes if num_nodes else 0.0
    energy_tx, energy_rx, energy_sleep = compute_energy_breakdown(metrics, num_nodes)
    collisions = int(metrics.get("collisions", 0))

    energy_class = {
        key: metrics[key]
        for key in metrics
        if key.startswith("energy_class_")
    }

    result: dict[str, object] = {
        "class": task["class"],
        "interval_s": task["interval_s"],
        "replicate": task["replicate"],
        "pdr": float(metrics.get("PDR", 0.0)),
        "energy_per_node_J": energy_per_node,
        "energy_tx_J": energy_tx,
        "energy_rx_J": energy_rx,
        "energy_sleep_J": energy_sleep,
        "collisions": float(collisions),
        "avg_delay_s": float(metrics.get("avg_delay_s", 0.0)),
        "_log_energy_class": energy_class,
    }
    return result


def main() -> None:  # noqa: D401 - CLI entry point
    parser = argparse.ArgumentParser(
        description="Run class load simulations for LoRaWAN classes A/B/C"
    )
    parser.add_argument("--nodes", type=positive_int, default=50, help="Number of nodes")
    parser.add_argument(
        "--interval",
        type=float,
        help="Single packet interval in seconds (additional to any list values)",
    )
    parser.add_argument(
        "--interval-list",
        type=float,
        action="append",
        help=(
            "Packet interval list in seconds. May be repeated. "
            "Defaults to 60, 300, 900 s when omitted."
        ),
    )
    parser.add_argument(
        "--packets", type=positive_int, default=40, help="Packets to send per node"
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=5,
        help="Number of replicates for each class/interval combination",
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
        "--resume",
        action="store_true",
        help="Skip simulations that already exist in the detailed CSV",
    )
    add_worker_argument(parser, default="auto")
    add_execution_profile_argument(parser)
    args = parser.parse_args()

    configure_logging(args.verbose)

    profile = resolve_execution_profile(args.profile)
    user_supplied_intervals = args.interval is not None or bool(args.interval_list)
    intervals = parse_intervals(args.interval, args.interval_list)
    if profile == "ci":
        if user_supplied_intervals:
            intervals = intervals[:1]
        else:
            intervals = CI_INTERVALS.copy()

    replicates = args.replicates if profile != "ci" else CI_REPLICATES
    num_nodes = args.nodes if profile != "ci" else min(args.nodes, CI_NODES)
    packets = args.packets if profile != "ci" else min(args.packets, CI_PACKETS)

    classes = ["A", "B", "C"]
    summary_rows: list[dict[str, float | str]] = []

    tasks: list[dict[str, object]] = []
    for class_type in classes:
        LOGGER.info("=== Simulating class %s ===", class_type)
        for interval_s in intervals:
            LOGGER.info("Interval %.1f s", interval_s)
            for replicate in range(1, replicates + 1):
                seed = args.seed + replicate - 1
                tasks.append(
                    {
                        "class": class_type,
                        "interval_s": interval_s,
                        "replicate": replicate,
                        "seed": seed,
                        "nodes": num_nodes,
                        "gateways": args.gateway,
                        "packets": packets,
                        "adr_node": args.adr_node,
                        "adr_server": args.adr_server,
                    }
                )

    if args.resume and RESULTS_PATH.exists():
        original_count = len(tasks)
        tasks = filter_completed_tasks(
            RESULTS_PATH, ("class", "interval_s", "replicate"), tasks
        )
        skipped = original_count - len(tasks)
        LOGGER.info("Skipping %d previously completed task(s) thanks to --resume", skipped)

    worker_count = resolve_worker_count(args.workers, len(tasks))
    if worker_count > 1:
        LOGGER.info("Using %d worker processes", worker_count)

    def log_progress(task: dict[str, object], result: dict[str, object], index: int) -> None:
        energy_class = result.get("_log_energy_class", {})
        LOGGER.info(
            (
                "Replicate %d -> PDR %.3f, collisions %.0f, energy/node %.4f J "
                "(tx %.4f J, rx %.4f J, sleep %.4f J), class energy %s"
            ),
            task["replicate"],
            result["pdr"],
            result["collisions"],
            result["energy_per_node_J"],
            result["energy_tx_J"],
            result["energy_rx_J"],
            result["energy_sleep_J"],
            energy_class,
        )

    results = execute_simulation_tasks(
        tasks,
        run_single_simulation,
        max_workers=worker_count,
        progress_callback=log_progress,
    )

    for row in results:
        row.pop("_log_energy_class", None)

    fieldnames = [
        "class",
        "interval_s",
        "replicate",
        "pdr",
        "pdr_mean",
        "pdr_std",
        "energy_per_node_J",
        "energy_tx_J",
        "energy_rx_J",
        "energy_sleep_J",
        "collisions",
        "avg_delay_s",
    ]
    summary_entries = summarise_metrics(
        results,
        ["class", "interval_s"],
        [
            "pdr",
            "energy_per_node_J",
            "energy_tx_J",
            "energy_rx_J",
            "energy_sleep_J",
            "collisions",
            "avg_delay_s",
        ],
    )
    summary_lookup = {
        (entry["class"], entry["interval_s"]): entry for entry in summary_entries
    }

    for row in results:
        summary = summary_lookup.get((row["class"], row["interval_s"]))
        if summary:
            row["pdr_mean"] = summary["pdr_mean"]
            row["pdr_std"] = summary["pdr_std"]

    write_csv(RESULTS_PATH, fieldnames, results)

    for class_type in classes:
        for interval_s in intervals:
            entry = summary_lookup.get((class_type, interval_s))
            if entry:
                summary_rows.append(entry)

    print(f"Results saved to {RESULTS_PATH}")
    print()
    print("Summary (per class and interval):")
    header = (
        f"{'Class':<7}{'Interval [s]':>14}{'PDR mean':>12}{'PDR std':>10}"
        f"{'Energy/node [J]':>18}{'Tx [J]':>10}{'Rx [J]':>10}{'Sleep [J]':>12}"
        f"{'Collisions':>12}{'Avg delay [s]':>16}"
    )
    print(header)
    print("-" * len(header))
    for entry in summary_rows:
        print(
            f"{entry['class']:<7}{entry['interval_s']:>14.1f}{entry['pdr_mean']:>12.3f}"
            f"{entry['pdr_std']:>10.3f}{entry['energy_per_node_J_mean']:>18.4f}"
            f"{entry['energy_tx_J_mean']:>10.4f}{entry['energy_rx_J_mean']:>10.4f}"
            f"{entry['energy_sleep_J_mean']:>12.4f}{entry['collisions_mean']:>12.2f}"
            f"{entry['avg_delay_s_mean']:>16.3f}"
        )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
