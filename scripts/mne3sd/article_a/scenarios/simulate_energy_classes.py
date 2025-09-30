#!/usr/bin/env python3
"""Simulate the energy footprint of LoRaWAN termination classes.

The script sweeps different LoRaWAN classes (A/B/C), duty-cycle limits and
MAC options and stores detailed energy statistics in
``results/mne3sd/article_a/energy_consumption.csv``.  Each configuration can be
replicated multiple times to capture stochastic variability.

Example
-------

.. code-block:: console

   python -m scripts.mne3sd.article_a.scenarios.simulate_energy_classes \\
       --nodes 40 --packets 20 --duty-cycle 0.01 --duty-cycle 0.001 \\
       --classes A --classes C --replicates 5 --seed 7 --mode random

The resulting CSV contains one row per replicate and includes the average
energy per node, a breakdown of the transmit/receive/sleep contributions and
the energy spent per successfully delivered message.  A secondary
``energy_consumption_summary.csv`` file aggregates the mean and population
standard deviation for the main metrics across replicates.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

# Allow running the script from a clone without installation
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)

from loraflexsim.launcher import Simulator  # noqa: E402
from scripts.mne3sd.common import (  # noqa: E402
    add_execution_profile_argument,
    add_worker_argument,
    execute_simulation_tasks,
    filter_completed_tasks,
    resolve_execution_profile,
    resolve_worker_count,
    summarise_metrics,
    write_csv,
)


LOGGER = logging.getLogger("simulate_energy_classes")
ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "results" / "mne3sd" / "article_a"
DETAIL_CSV = RESULTS_DIR / "energy_consumption.csv"
SUMMARY_CSV = RESULTS_DIR / "energy_consumption_summary.csv"

# Conservative defaults mirroring previous Article A experiments
DEFAULT_CLASSES = ("A", "B", "C")
DEFAULT_DUTY_CYCLES = (0.01,)
DEFAULT_INTERVAL_S = 300.0
DEFAULT_PACKETS = 40
DEFAULT_REPLICATES = 5

# Lightweight configuration for continuous integration runs
CI_NODES = 10
CI_PACKETS = 5
CI_REPLICATES = 1
CI_CLASSES = ("A",)
CI_DUTY_CYCLES = (0.01,)

FAST_NODE_CAP = 150
FAST_PACKETS = 20
FAST_REPLICATES = 3


def positive_int(value: str) -> int:
    """Return ``value`` converted to a strictly positive integer."""

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def positive_float(value: str) -> float:
    """Return ``value`` converted to a strictly positive float."""

    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return fvalue


def parse_class_list(values: Iterable[str] | None) -> list[str]:
    """Validate and normalise the list of LoRaWAN classes to simulate."""

    if not values:
        return list(DEFAULT_CLASSES)
    classes: list[str] = []
    for candidate in values:
        normalised = candidate.strip().upper()
        if normalised not in {"A", "B", "C"}:
            raise argparse.ArgumentTypeError(
                f"unsupported class '{candidate}', expected A, B or C"
            )
        if normalised not in classes:
            classes.append(normalised)
    return classes


def compute_energy_breakdown(
    metrics: dict, num_nodes: int
) -> tuple[float, float, float]:
    """Return per-node TX, RX and sleep energy from simulator metrics."""

    if num_nodes <= 0:
        return 0.0, 0.0, 0.0

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
    scale = 1.0 / num_nodes
    return total_tx * scale, total_rx * scale, total_sleep * scale


def iter_configurations(
    classes: Iterable[str],
    duty_cycles: Iterable[float],
    replicates: int,
) -> Iterator[tuple[str, float, int]]:
    """Yield ``(class, duty_cycle, replicate)`` combinations."""

    index = 0
    for class_type in classes:
        for duty_cycle in duty_cycles:
            for replicate in range(1, replicates + 1):
                yield class_type, duty_cycle, replicate
                index += 1


def configure_logging(verbose: bool) -> None:
    """Configure logging verbosity for the script."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def run_single_configuration(task: dict[str, object]) -> dict[str, object]:
    """Execute a single class/duty-cycle replicate and return the collected metrics."""

    sim = Simulator(
        num_nodes=int(task["nodes"]),
        packets_to_send=int(task["packets"]),
        packet_interval=float(task["interval_s"]),
        transmission_mode=str(task["mode"]),
        duty_cycle=float(task["duty_cycle"]),
        node_class=str(task["class"]),
        adr_node=bool(task["adr_node"]),
        adr_server=bool(task["adr_server"]),
        seed=int(task["seed"]),
    )
    sim.run()
    metrics = sim.get_metrics()

    nodes = int(task["nodes"])
    energy_nodes = float(metrics.get("energy_nodes_J", 0.0))
    delivered = int(metrics.get("delivered", 0))
    pdr = float(metrics.get("PDR", 0.0))
    avg_delay = float(metrics.get("avg_delay_s", 0.0))
    collisions = int(metrics.get("collisions", 0))
    energy_per_node = energy_nodes / nodes if nodes else 0.0
    energy_per_message = energy_nodes / delivered if delivered > 0 else 0.0
    energy_tx, energy_rx, energy_sleep = compute_energy_breakdown(metrics, nodes)
    energy_class_total = float(metrics.get(f"energy_class_{task['class']}_J", energy_nodes))

    return {
        "class": task["class"],
        "duty_cycle": task["duty_cycle_value"],
        "nodes": nodes,
        "packets": task["packets"],
        "interval_s": task["interval_s"],
        "mode": task["mode_label"],
        "replicate": task["replicate"],
        "seed": task["seed"],
        "adr_node": int(task["adr_node"]),
        "adr_server": int(task["adr_server"]),
        "pdr": pdr,
        "delivered": delivered,
        "collisions": collisions,
        "avg_delay_s": avg_delay,
        "energy_per_node_J": energy_per_node,
        "energy_per_message_J": energy_per_message,
        "energy_tx_per_node_J": energy_tx,
        "energy_rx_per_node_J": energy_rx,
        "energy_sleep_per_node_J": energy_sleep,
        "energy_class_total_J": energy_class_total,
    }


def main() -> None:  # noqa: D401 - CLI entry point
    parser = argparse.ArgumentParser(
        description="Simulate energy consumption for LoRaWAN classes"
    )
    parser.add_argument("--nodes", type=positive_int, default=50, help="Number of nodes")
    parser.add_argument(
        "--packets",
        type=positive_int,
        default=DEFAULT_PACKETS,
        help="Packets to send per node",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=DEFAULT_INTERVAL_S,
        help="Mean packet interval in seconds",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "periodic"],
        default="random",
        help="Traffic generation mode",
    )
    parser.add_argument(
        "--classes",
        action="append",
        help="LoRaWAN class to include (A, B or C). May be repeated.",
    )
    parser.add_argument(
        "--duty-cycle",
        action="append",
        type=positive_float,
        help="Duty-cycle constraint as a fraction (e.g. 0.01 for 1%%).",
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=DEFAULT_REPLICATES,
        help="Number of stochastic replicates per configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base random seed used for the first replicate",
    )
    parser.add_argument(
        "--adr-node",
        action="store_true",
        help="Enable ADR on the nodes (MAC parameter)",
    )
    parser.add_argument(
        "--adr-server",
        action="store_true",
        help="Enable ADR on the server (MAC parameter)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip simulations that already exist in the detailed CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    add_worker_argument(parser, default="auto")
    add_execution_profile_argument(parser)
    args = parser.parse_args()

    configure_logging(args.verbose)

    profile = resolve_execution_profile(args.profile)
    classes = parse_class_list(args.classes)
    duty_cycles = [float(dc) for dc in (args.duty_cycle or DEFAULT_DUTY_CYCLES)]

    nodes = args.nodes
    packets = args.packets
    replicates = args.replicates
    if profile == "ci":
        classes = list(CI_CLASSES)
        duty_cycles = list(CI_DUTY_CYCLES)
        nodes = min(nodes, CI_NODES)
        packets = min(packets, CI_PACKETS)
        replicates = min(replicates, CI_REPLICATES)
    elif profile == "fast":
        nodes = min(nodes, FAST_NODE_CAP)
        packets = min(packets, FAST_PACKETS)
        replicates = min(replicates, FAST_REPLICATES)

    LOGGER.info(
        "Simulating energy consumption for classes %s with duty cycles %s",
        ", ".join(classes),
        ", ".join(f"{dc:.4f}" for dc in duty_cycles),
    )

    tasks: list[dict[str, object]] = []
    seed_counter = args.seed
    for class_type, duty_cycle, replicate in iter_configurations(
        classes, duty_cycles, replicates
    ):
        LOGGER.info(
            "Class %s, duty cycle %.4f (replicate %d/%d)",
            class_type,
            duty_cycle,
            replicate,
            replicates,
        )
        tasks.append(
            {
                "class": class_type,
                "duty_cycle": duty_cycle,
                "duty_cycle_value": duty_cycle,
                "nodes": nodes,
                "packets": packets,
                "interval_s": args.interval,
                "mode": args.mode.capitalize(),
                "mode_label": args.mode,
                "replicate": replicate,
                "seed": seed_counter,
                "adr_node": args.adr_node,
                "adr_server": args.adr_server,
            }
        )
        seed_counter += 1

    if args.resume and DETAIL_CSV.exists():
        original_count = len(tasks)
        tasks = filter_completed_tasks(
            DETAIL_CSV, ("class", "duty_cycle", "replicate"), tasks
        )
        skipped = original_count - len(tasks)
        LOGGER.info("Skipping %d previously completed task(s) thanks to --resume", skipped)

    worker_count = resolve_worker_count(args.workers, len(tasks))
    if worker_count > 1:
        LOGGER.info("Using %d worker processes", worker_count)

    def log_progress(task: dict[str, object], result: dict[str, object], index: int) -> None:
        LOGGER.debug(
            "Result -> PDR %.3f, collisions %d, energy/node %.4f J (tx %.4f J, rx %.4f J, sleep %.4f J)",
            result["pdr"],
            result["collisions"],
            result["energy_per_node_J"],
            result["energy_tx_per_node_J"],
            result["energy_rx_per_node_J"],
            result["energy_sleep_per_node_J"],
        )

    rows = execute_simulation_tasks(
        tasks,
        run_single_configuration,
        max_workers=worker_count,
        progress_callback=log_progress,
    )

    if not rows:
        LOGGER.warning("No simulations executed; aborting without writing CSV files")
        return

    write_csv(
        DETAIL_CSV,
        [
            "class",
            "duty_cycle",
            "nodes",
            "packets",
            "interval_s",
            "mode",
            "replicate",
            "seed",
            "adr_node",
            "adr_server",
            "pdr",
            "delivered",
            "collisions",
            "avg_delay_s",
            "energy_per_node_J",
            "energy_per_message_J",
            "energy_tx_per_node_J",
            "energy_rx_per_node_J",
            "energy_sleep_per_node_J",
            "energy_class_total_J",
        ],
        rows,
    )

    summary_rows = summarise_metrics(
        rows,
        ["class", "duty_cycle", "mode", "adr_node", "adr_server"],
        [
            "pdr",
            "avg_delay_s",
            "energy_per_node_J",
            "energy_per_message_J",
            "energy_tx_per_node_J",
            "energy_rx_per_node_J",
            "energy_sleep_per_node_J",
            "collisions",
        ],
    )
    write_csv(SUMMARY_CSV, summary_rows[0].keys(), summary_rows)

    LOGGER.info("Saved %s and %s", DETAIL_CSV, SUMMARY_CSV)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
