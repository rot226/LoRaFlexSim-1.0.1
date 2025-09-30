"""Compute RSSI/SNR margins for long-range presets at specific distances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from loraflexsim.launcher import Channel, Simulator
from loraflexsim.scenarios.long_range import LONG_RANGE_DISTANCES, LONG_RANGE_RECOMMENDATIONS


def _loss_model_for_preset(preset: str) -> str:
    return "hata" if preset == "flora_hata" else "lognorm"


def evaluate_margin(
    preset: str,
    distances_km: Iterable[float],
    *,
    tx_power_dBm: float,
    tx_gain_dB: float,
    rx_gain_dB: float,
    cable_loss_dB: float,
    sf: int,
    bandwidth: int,
) -> list[dict[str, float]]:
    channel = Channel(environment=preset, flora_loss_model=_loss_model_for_preset(preset))
    channel.tx_antenna_gain_dB = tx_gain_dB
    channel.rx_antenna_gain_dB = rx_gain_dB
    channel.cable_loss_dB = cable_loss_dB
    channel.bandwidth = bandwidth

    sensitivity = Channel.FLORA_SENSITIVITY[sf][bandwidth]
    required_snr = Simulator.REQUIRED_SNR[sf]
    results: list[dict[str, float]] = []

    for distance_km in distances_km:
        distance_m = distance_km * 1000.0
        rssi, snr = channel.compute_rssi(tx_power_dBm, distance_m, sf)
        results.append(
            {
                "distance_km": distance_km,
                "rssi_dBm": rssi,
                "snr_dB": snr,
                "sensitivity_dBm": sensitivity,
                "rssi_margin_dB": rssi - sensitivity,
                "snr_margin_dB": snr - required_snr,
            }
        )
    return results


def _default_distances(preset: str) -> list[float]:
    params = LONG_RANGE_RECOMMENDATIONS[preset]
    distances = params.distances or tuple(LONG_RANGE_DISTANCES)
    # Surface the most representative distances in km.
    return sorted({round(d / 1000.0, 1) for d in distances} | {10.0, 12.0, 15.0})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate RSSI/SNR margins for long-range presets.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(LONG_RANGE_RECOMMENDATIONS),
        default="flora_hata",
        help="Long-range preset to analyse (defaults to flora_hata).",
    )
    parser.add_argument(
        "--tx-power",
        type=float,
        help="Transmit power in dBm (defaults to preset recommendation).",
    )
    parser.add_argument(
        "--tx-gain",
        type=float,
        help="Gateway/node antenna gain in dBi (defaults to preset recommendation).",
    )
    parser.add_argument(
        "--rx-gain",
        type=float,
        help="Gateway receive gain in dBi (defaults to preset recommendation).",
    )
    parser.add_argument(
        "--cable-loss",
        type=float,
        help="Cable loss in dB (defaults to preset recommendation).",
    )
    parser.add_argument(
        "--sf",
        type=int,
        default=12,
        help="LoRa spreading factor to evaluate (default: 12).",
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=125_000,
        help="Bandwidth in Hz (default: 125000).",
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        help="Distances to evaluate in kilometres (defaults to preset distances + 10/12/15 km).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to export the results as CSV.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    params = LONG_RANGE_RECOMMENDATIONS[args.preset]
    tx_power = args.tx_power if args.tx_power is not None else params.tx_power_dBm
    tx_gain = args.tx_gain if args.tx_gain is not None else params.tx_antenna_gain_dB
    rx_gain = args.rx_gain if args.rx_gain is not None else params.rx_antenna_gain_dB
    cable_loss = args.cable_loss if args.cable_loss is not None else params.cable_loss_dB

    distances = args.distances or _default_distances(args.preset)
    results = evaluate_margin(
        args.preset,
        distances,
        tx_power_dBm=tx_power,
        tx_gain_dB=tx_gain,
        rx_gain_dB=rx_gain,
        cable_loss_dB=cable_loss,
        sf=args.sf,
        bandwidth=args.bandwidth,
    )

    header = [
        "distance_km",
        "rssi_dBm",
        "snr_dB",
        "sensitivity_dBm",
        "rssi_margin_dB",
        "snr_margin_dB",
    ]
    print("Preset:", args.preset)
    print(
        f"TX power={tx_power:.1f} dBm, gains TX/RX={tx_gain:.1f}/{rx_gain:.1f} dBi, cable loss={cable_loss:.1f} dB",
    )
    print(f"SF={args.sf}, bandwidth={args.bandwidth} Hz")
    print("\nDistance (km)  RSSI (dBm)  Margin (dB)  SNR (dB)  SNR margin (dB)")
    for row in results:
        print(
            f"{row['distance_km']:>12.1f}  {row['rssi_dBm']:>10.1f}  {row['rssi_margin_dB']:>10.1f}"
            f"  {row['snr_dB']:>8.2f}  {row['snr_margin_dB']:>14.2f}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
