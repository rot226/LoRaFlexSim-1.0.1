#!/usr/bin/env python3
"""Compare RSSI/SNR against FLoRa references for SF7–SF12."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.obstacle_loss import ObstacleLoss
from loraflexsim.tests.reference_traces import _flora_rssi_snr


@dataclass(frozen=True)
class RegressionResult:
    sf: int
    profile: str
    distance_m: float
    rssi_sim: float
    rssi_ref: float
    snr_sim: float
    snr_ref: float

    @property
    def rssi_delta(self) -> float:
        return self.rssi_sim - self.rssi_ref

    @property
    def snr_delta(self) -> float:
        return self.snr_sim - self.snr_ref


def _flora_reference(
    channel: Channel,
    tx_power_dBm: float,
    distance_m: float,
    sf: int,
    *,
    obstacle: ObstacleLoss | None,
    tx_pos: tuple[float, float],
    rx_pos: tuple[float, float],
) -> tuple[float, float]:
    """Compute the RSSI/SNR expected from the FLoRa formulas."""

    rssi, snr = _flora_rssi_snr(channel, tx_power_dBm, distance_m, sf)

    if obstacle is not None:
        att = obstacle.loss(tx_pos, rx_pos)
        rssi -= att
        snr -= att

    return rssi, snr


def _build_channel(obstacle: ObstacleLoss | None) -> Channel:
    return Channel(
        phy_model="flora_full",
        environment="flora",
        shadowing_std=0.0,
        fast_fading_std=0.0,
        noise_floor_std=0.0,
        flora_capture=True,
        use_flora_curves=True,
        obstacle_loss=obstacle,
    )


def _iterate_results(obstacle: ObstacleLoss | None, profile: str) -> Iterable[RegressionResult]:
    distance_map = {7: 40.0, 8: 80.0, 9: 160.0, 10: 320.0, 11: 640.0, 12: 1280.0}
    tx_power = 14.0
    channel = _build_channel(obstacle)
    tx_pos = (0.0, 0.0)
    for sf in range(7, 13):
        distance = distance_map[sf]
        rx_pos = (distance, 0.0)
        rssi_sim, snr_sim = channel.compute_rssi(tx_power, distance, sf=sf, tx_pos=tx_pos, rx_pos=rx_pos)

        ref_channel = _build_channel(None)
        ref_channel.tx_antenna_gain_dB = channel.tx_antenna_gain_dB
        ref_channel.rx_antenna_gain_dB = channel.rx_antenna_gain_dB
        ref_channel.cable_loss_dB = channel.cable_loss_dB
        ref_channel.rssi_offset_dB = channel.rssi_offset_dB
        ref_channel.snr_offset_dB = channel.snr_offset_dB
        ref_channel.processing_gain = channel.processing_gain
        ref_channel.bandwidth = channel.bandwidth
        ref_channel.system_loss_dB = channel.system_loss_dB
        ref_rssi, ref_snr = _flora_reference(
            ref_channel,
            tx_power,
            distance,
            sf,
            obstacle=obstacle,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
        )

        yield RegressionResult(sf, profile, distance, rssi_sim, ref_rssi, snr_sim, ref_snr)


def run_regression(output: Path | None, tolerance: float = 0.6) -> int:
    obstacle = ObstacleLoss.from_raster([[4.0]], cell_size=10.0, material="concrete")
    results = list(_iterate_results(None, "clear"))
    results.extend(_iterate_results(obstacle, "obstacle"))

    header = (
        f"{'SF':>2}  {'Profile':>8}  {'Dist (m)':>8}  {'RSSI sim':>10}  {'RSSI ref':>10}  "
        f"{'ΔRSSI':>8}  {'SNR sim':>10}  {'SNR ref':>10}  {'ΔSNR':>8}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.sf:>2}  {res.profile:>8}  {res.distance_m:8.1f}  {res.rssi_sim:10.3f}  "
            f"{res.rssi_ref:10.3f}  {res.rssi_delta:8.3f}  {res.snr_sim:10.3f}  {res.snr_ref:10.3f}  {res.snr_delta:8.3f}"
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "sf",
                    "profile",
                    "distance_m",
                    "rssi_sim",
                    "rssi_ref",
                    "rssi_delta",
                    "snr_sim",
                    "snr_ref",
                    "snr_delta",
                ]
            )
            for res in results:
                writer.writerow(
                    [
                        res.sf,
                        res.profile,
                        f"{res.distance_m:.1f}",
                        f"{res.rssi_sim:.6f}",
                        f"{res.rssi_ref:.6f}",
                        f"{res.rssi_delta:.6f}",
                        f"{res.snr_sim:.6f}",
                        f"{res.snr_ref:.6f}",
                        f"{res.snr_delta:.6f}",
                    ]
                )

    max_rssi = max(abs(res.rssi_delta) for res in results)
    max_snr = max(abs(res.snr_delta) for res in results)
    if max_rssi > tolerance or max_snr > tolerance:
        print(
            f"Maximum deviation exceeds tolerance ({max_rssi:.3f} dB RSSI, {max_snr:.3f} dB SNR > {tolerance:.3f})",
        )
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path for the regression summary.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=12.0,
        help="Maximum accepted deviation in dB (default: 12).",
    )
    args = parser.parse_args()
    exit_code = run_regression(args.output, tolerance=args.tolerance)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
