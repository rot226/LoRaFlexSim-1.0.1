"""Validate the kilometre-scale long range scenario.

The helper loads the node and gateway placement defined in
``examples/long_range.yaml`` then executes a deterministic LoRaFlexSim
simulation. The script reports the resulting Packet Delivery Ratio (PDR)
and optionally checks that it meets a minimum threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/validate_long_range.py`` to import the package when run
# from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.channel import Channel

DEFAULT_SCENARIO = ROOT_DIR / "examples" / "long_range.yaml"


def run_validation(
    scenario_path: Path,
    *,
    interval_s: float = 1800.0,
    packets: int = 5,
    seed: int | None = 1,
    threshold: float | None = 0.95,
    area_size: float = 8000.0,
) -> float:
    """Run the long range validation and return the PDR as a fraction."""

    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    channel = Channel(
        path_loss_exp=2.08,
        shadowing_std=0.0,
        path_loss_d0=127.41,
        reference_distance=40.0,
        tx_antenna_gain_dB=6.0,
        rx_antenna_gain_dB=8.0,
    )

    simulator = Simulator(
        transmission_mode="Periodic",
        packet_interval=interval_s,
        first_packet_interval=interval_s,
        packets_to_send=packets,
        mobility=False,
        duty_cycle=None,
        channels=[channel],
        area_size=area_size,
        config_file=str(scenario_path),
        seed=seed,
    )
    simulator.run()
    metrics = simulator.get_metrics()
    pdr = metrics["PDR"]

    delivered = int(metrics["delivered"])
    attempted = int(metrics["tx_attempted"])
    pct = pdr * 100.0
    print(
        f"PDR={pct:.2f}% ({delivered} packets delivered over {attempted} attempts)"
    )

    if threshold is not None:
        threshold_pct = threshold * 100.0
        if pdr < threshold:
            raise SystemExit(
                f"PDR {pct:.2f}% is below the required threshold of {threshold_pct:.2f}%"
            )
        print(f"Threshold of {threshold_pct:.2f}% satisfied.")

    return pdr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="Path to the YAML scenario file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1800.0,
        help="Interval between transmissions in seconds",
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=5,
        help="Packets to send per node for the validation run",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum acceptable PDR (fraction between 0 and 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed forwarded to the simulator RNGs",
    )
    parser.add_argument(
        "--area-size",
        type=float,
        default=8000.0,
        help="Side length of the simulated area in metres",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_validation(
        args.scenario,
        interval_s=args.interval,
        packets=args.packets,
        seed=args.seed,
        threshold=args.threshold,
        area_size=args.area_size,
    )

if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
