"""Benchmark energy consumption for LoRaWAN classes A/B/C."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from loraflexsim.launcher.simulator import Simulator

DEFAULT_OUTPUT = Path("results/energy_classes.csv")


def run_benchmark(
    *,
    nodes: int,
    gateways: int,
    area_size: float,
    packet_interval: float,
    packets_to_send: int,
    mode: str,
    seed: int,
    duty_cycle: float | None,
    output: Path,
) -> Path:
    """Run the benchmark for each LoRaWAN class and export a CSV report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "class",
        "pdr",
        "energy_nodes_J",
        "energy_per_node_J",
        "energy_tx_J",
        "energy_rx_J",
        "energy_sleep_J",
        "energy_ramp_J",
        "energy_startup_J",
        "energy_preamble_J",
        "energy_processing_J",
        "energy_listen_J",
    ]
    rows: list[dict[str, float | str]] = []
    for cls in ("A", "B", "C"):
        sim = Simulator(
            num_nodes=nodes,
            num_gateways=gateways,
            area_size=area_size,
            transmission_mode=mode,
            packet_interval=packet_interval,
            packets_to_send=packets_to_send,
            duty_cycle=duty_cycle,
            mobility=False,
            node_class=cls,
            seed=seed,
        )
        sim.run()
        metrics = sim.get_metrics()
        per_node = metrics["energy_nodes_J"] / nodes if nodes > 0 else 0.0
        breakdown = _aggregate_states(metrics["energy_breakdown_by_node"].values())
        rows.append(
            {
                "class": cls,
                "pdr": metrics["PDR"],
                "energy_nodes_J": metrics["energy_nodes_J"],
                "energy_per_node_J": per_node,
                "energy_tx_J": breakdown.get("tx", 0.0),
                "energy_rx_J": breakdown.get("rx", 0.0),
                "energy_sleep_J": breakdown.get("sleep", 0.0),
                "energy_ramp_J": breakdown.get("ramp", 0.0),
                "energy_startup_J": breakdown.get("startup", 0.0),
                "energy_preamble_J": breakdown.get("preamble", 0.0),
                "energy_processing_J": breakdown.get("processing", 0.0),
                "energy_listen_J": breakdown.get("listen", 0.0),
            }
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _aggregate_states(breakdowns: Iterable[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in breakdowns:
        for state, value in entry.items():
            totals[state] = totals.get(state, 0.0) + value
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate energy benchmarks for LoRaWAN classes A/B/C",
    )
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--gateways", type=int, default=1, help="Number of gateways")
    parser.add_argument(
        "--area",
        type=float,
        default=1000.0,
        help="Deployment area size (meters)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Packet interval in seconds",
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=10,
        help="Packets to send per node (0 for continuous mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["Periodic", "Random"],
        default="Periodic",
        help="Transmission mode",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base RNG seed")
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=0.01,
        help="Maximum duty cycle (set to 0 to disable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    duty_cycle = None if args.duty_cycle == 0 else args.duty_cycle
    return run_benchmark(
        nodes=args.nodes,
        gateways=args.gateways,
        area_size=args.area,
        packet_interval=args.interval,
        packets_to_send=args.packets,
        mode=args.mode,
        seed=args.seed,
        duty_cycle=duty_cycle,
        output=args.output,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
