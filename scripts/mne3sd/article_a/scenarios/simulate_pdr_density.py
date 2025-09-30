#!/usr/bin/env python3
"""Analyse how gateway density and node population impact PDR.

The script explores different base-station densities (gateways per km²), node
populations and spreading-factor strategies (adaptive ADR versus fixed SF) and
records the resulting delivery metrics to
``results/mne3sd/article_a/pdr_density.csv``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from itertools import product
from pathlib import Path
from typing import Iterable

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


LOGGER = logging.getLogger("simulate_pdr_density")
ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "results" / "mne3sd" / "article_a"
DETAIL_CSV = RESULTS_DIR / "pdr_density.csv"
SUMMARY_CSV = RESULTS_DIR / "pdr_density_summary.csv"

DEFAULT_DENSITIES = (0.25, 0.5, 1.0)  # gateways per km²
DEFAULT_NODE_COUNTS = (50, 100, 200)
DEFAULT_SF_MODES = ("adaptive", "fixed7", "fixed9", "fixed12")
DEFAULT_REPLICATES = 5

CI_DENSITIES = (0.5,)
CI_NODE_COUNTS = (50,)
CI_SF_MODES = ("adaptive",)
CI_REPLICATES = 1

FAST_NODE_CAP = 150
FAST_REPLICATES = 3
FAST_PACKETS = 10


def positive_float(value: str) -> float:
    """Return ``value`` converted to a strictly positive float."""

    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return fvalue


def positive_int(value: str) -> int:
    """Return ``value`` converted to a strictly positive integer."""

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def parse_density(values: Iterable[str] | None) -> list[float]:
    """Parse gateway density values expressed in gateways/km²."""

    if not values:
        return list(DEFAULT_DENSITIES)
    densities: list[float] = []
    for value in values:
        density = positive_float(value)
        if density not in densities:
            densities.append(density)
    return densities


def parse_node_counts(values: Iterable[str] | None) -> list[int]:
    """Parse the desired node counts."""

    if not values:
        return list(DEFAULT_NODE_COUNTS)
    counts: list[int] = []
    for value in values:
        count = positive_int(value)
        if count not in counts:
            counts.append(count)
    return counts


def parse_sf_modes(values: Iterable[str] | None) -> list[str]:
    """Return the spreading-factor strategies to evaluate."""

    if not values:
        return list(DEFAULT_SF_MODES)
    modes: list[str] = []
    for value in values:
        cleaned = value.strip().lower()
        if cleaned == "adaptive":
            label = "adaptive"
        elif cleaned.startswith("fixed"):
            try:
                sf = int(cleaned.replace("fixed", ""))
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise argparse.ArgumentTypeError(
                    f"Invalid fixed SF specification '{value}'"
                ) from exc
            if sf < 7 or sf > 12:
                raise argparse.ArgumentTypeError(
                    f"Invalid spreading factor '{sf}', expected 7–12"
                )
            label = f"fixed{sf}"
        else:
            raise argparse.ArgumentTypeError(
                f"Unsupported SF mode '{value}'. Use 'adaptive' or 'fixed<SF>'."
            )
        if label not in modes:
            modes.append(label)
    return modes


def configure_logging(verbose: bool) -> None:
    """Initialise the logging subsystem."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def gateways_from_density(density: float, area_km2: float) -> int:
    """Return the number of gateways covering ``area_km2`` at ``density``."""

    gateways = max(1, round(density * area_km2))
    return gateways


def area_side_from_surface(area_km2: float) -> float:
    """Return the side length in metres of a square covering ``area_km2``."""

    return math.sqrt(area_km2 * 1_000_000.0)


def run_single_configuration(task: dict[str, object]) -> dict[str, object]:
    """Execute a single density/node/SF replicate and return the collected metrics."""

    gateways = int(task["gateways"])
    node_count = int(task["nodes"])
    sim = Simulator(
        num_nodes=node_count,
        num_gateways=gateways,
        area_size=float(task["area_side_m"]),
        packets_to_send=int(task["packets"]),
        packet_interval=float(task["interval_s"]),
        transmission_mode="Random",
        seed=int(task["seed"]),
        **task["sf_kwargs"],
    )
    sim.run()
    metrics = sim.get_metrics()

    delivered = int(metrics.get("delivered", 0))
    collisions = int(metrics.get("collisions", 0))
    pdr = float(metrics.get("PDR", 0.0))
    avg_delay = float(metrics.get("avg_delay_s", 0.0))
    throughput = float(metrics.get("throughput_bps", 0.0))
    pdr_by_sf = metrics.get("pdr_by_sf", {})
    pdr_by_class = metrics.get("pdr_by_class", {})
    energy_nodes = float(metrics.get("energy_nodes_J", 0.0))
    energy_per_node = energy_nodes / node_count if node_count else 0.0

    row = {
        "area_km2": task["area_km2"],
        "density_gw_per_km2": task["density"],
        "gateways": gateways,
        "nodes": node_count,
        "sf_mode": task["sf_mode"],
        "packets": task["packets"],
        "interval_s": task["interval_s"],
        "replicate": task["replicate"],
        "seed": task["seed"],
        "pdr": pdr,
        "delivered": delivered,
        "collisions": collisions,
        "avg_delay_s": avg_delay,
        "throughput_bps": throughput,
        "energy_per_node_J": energy_per_node,
    }

    for sf in range(7, 13):
        row[f"pdr_sf{sf}"] = float(pdr_by_sf.get(sf, 0.0))

    for class_type, class_pdr in pdr_by_class.items():
        row[f"pdr_class_{class_type}"] = float(class_pdr)

    return row


def main() -> None:  # noqa: D401 - CLI entry point
    parser = argparse.ArgumentParser(
        description="Simulate PDR as a function of base-station density"
    )
    parser.add_argument(
        "--area-km2",
        type=positive_float,
        default=4.0,
        help="Deployment area in square kilometres",
    )
    parser.add_argument(
        "--density",
        action="append",
        help="Gateway density in gateways per km^2. May be repeated.",
    )
    parser.add_argument(
        "--nodes",
        action="append",
        help="Number of nodes to simulate. May be repeated.",
    )
    parser.add_argument(
        "--sf-mode",
        action="append",
        help="Spreading-factor mode: 'adaptive' or 'fixed<SF>'. May be repeated.",
    )
    parser.add_argument(
        "--packets",
        type=positive_int,
        default=20,
        help="Packets transmitted per node",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=300.0,
        help="Mean packet interval in seconds",
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=DEFAULT_REPLICATES,
        help="Replicates per configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base random seed used for the first replicate",
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
    densities = parse_density(args.density)
    node_counts = parse_node_counts(args.nodes)
    sf_modes = parse_sf_modes(args.sf_mode)

    area_km2 = args.area_km2
    packets = args.packets
    interval_s = args.interval
    replicates = args.replicates
    if profile == "ci":
        densities = list(CI_DENSITIES)
        node_counts = list(CI_NODE_COUNTS)
        sf_modes = list(CI_SF_MODES)
        replicates = min(replicates, CI_REPLICATES)
        packets = min(packets, 5)
    elif profile == "fast":
        replicates = min(replicates, FAST_REPLICATES)
        packets = min(packets, FAST_PACKETS)
        clamped_nodes: list[int] = []
        seen_nodes: set[int] = set()
        for count in node_counts:
            limited = min(count, FAST_NODE_CAP)
            if limited not in seen_nodes:
                clamped_nodes.append(limited)
                seen_nodes.add(limited)
        node_counts = clamped_nodes

    side_m = area_side_from_surface(area_km2)
    LOGGER.info(
        "Simulating area %.2f km² (side %.0f m) for densities %s and node counts %s",
        area_km2,
        side_m,
        ", ".join(f"{d:.2f}" for d in densities),
        ", ".join(str(n) for n in node_counts),
    )

    tasks: list[dict[str, object]] = []
    seed_counter = args.seed
    for density, node_count, sf_mode in product(densities, node_counts, sf_modes):
        gateways = gateways_from_density(density, area_km2)
        if sf_mode == "adaptive":
            sf_kwargs = {"fixed_sf": None, "adr_node": True, "adr_server": True}
        else:
            sf_value = int(sf_mode.replace("fixed", ""))
            sf_kwargs = {"fixed_sf": sf_value, "adr_node": False, "adr_server": False}

        LOGGER.info(
            "Density %.2f gw/km² (%d gateways), nodes %d, SF mode %s",
            density,
            gateways,
            node_count,
            sf_mode,
        )

        for replicate in range(1, replicates + 1):
            tasks.append(
                {
                    "area_km2": area_km2,
                    "area_side_m": side_m,
                    "density": density,
                    "density_gw_per_km2": density,
                    "gateways": gateways,
                    "nodes": node_count,
                    "sf_mode": sf_mode,
                    "sf_kwargs": dict(sf_kwargs),
                    "packets": packets,
                    "interval_s": interval_s,
                    "replicate": replicate,
                    "seed": seed_counter,
                }
            )
            seed_counter += 1

    if args.resume and DETAIL_CSV.exists():
        original_count = len(tasks)
        tasks = filter_completed_tasks(
            DETAIL_CSV,
            ("density_gw_per_km2", "nodes", "sf_mode", "replicate"),
            tasks,
        )
        skipped = original_count - len(tasks)
        LOGGER.info("Skipping %d previously completed task(s) thanks to --resume", skipped)

    worker_count = resolve_worker_count(args.workers, len(tasks))
    if worker_count > 1:
        LOGGER.info("Using %d worker processes", worker_count)

    def log_progress(task: dict[str, object], result: dict[str, object], index: int) -> None:
        LOGGER.debug(
            "Replicate %d -> PDR %.3f, collisions %d, avg delay %.3fs",
            task["replicate"],
            result["pdr"],
            result["collisions"],
            result["avg_delay_s"],
        )

    results = execute_simulation_tasks(
        tasks,
        run_single_configuration,
        max_workers=worker_count,
        progress_callback=log_progress,
    )

    if not results:
        LOGGER.warning("No simulations executed; skipping CSV generation")
        return

    fieldnames = [
        "area_km2",
        "density_gw_per_km2",
        "gateways",
        "nodes",
        "sf_mode",
        "packets",
        "interval_s",
        "replicate",
        "seed",
        "pdr",
        "delivered",
        "collisions",
        "avg_delay_s",
        "throughput_bps",
        "energy_per_node_J",
    ]
    fieldnames.extend([f"pdr_sf{sf}" for sf in range(7, 13)])
    class_columns = sorted(
        {key for row in results for key in row if key.startswith("pdr_class_")}
    )
    fieldnames.extend(class_columns)

    write_csv(DETAIL_CSV, fieldnames, results)

    summary_rows = summarise_metrics(
        results,
        ["density_gw_per_km2", "nodes", "sf_mode"],
        [
            "pdr",
            "avg_delay_s",
            "collisions",
            "throughput_bps",
            "energy_per_node_J",
            "pdr_sf7",
            "pdr_sf8",
            "pdr_sf9",
            "pdr_sf10",
            "pdr_sf11",
            "pdr_sf12",
        ],
    )
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    if summary_fieldnames:
        write_csv(SUMMARY_CSV, summary_fieldnames, summary_rows)
        LOGGER.info("Saved %s and %s", DETAIL_CSV, SUMMARY_CSV)
    else:
        LOGGER.info("Saved %s", DETAIL_CSV)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
