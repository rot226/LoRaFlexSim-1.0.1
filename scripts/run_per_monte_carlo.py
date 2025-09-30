#!/usr/bin/env python3
"""Monte-Carlo campaign comparing FLoRa logistic and Croce PER models."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.flora_phy import FloraPHY
@dataclass(frozen=True)
class MonteCarloResult:
    sf: int
    payload: int
    snr: float
    model: str
    per_theory: float
    per_mc: float

    @property
    def delta(self) -> float:
        return self.per_mc - self.per_theory


def _snr_range(start: float, stop: float, step: float) -> Iterable[float]:
    count = int(round((stop - start) / step))
    for idx in range(count + 1):
        yield round(start + idx * step, 6)


def _simulate(per: float, samples: int, rng: random.Random) -> float:
    if per <= 0.0:
        return 0.0
    if per >= 1.0:
        return 1.0
    failures = sum(1 for _ in range(samples) if rng.random() < per)
    return failures / samples


def run_campaign(
    sfs: Iterable[int],
    payloads: Iterable[int],
    snr_start: float,
    snr_stop: float,
    snr_step: float,
    samples: int,
    seed: int,
    output: Path | None,
) -> None:
    rng = random.Random(seed)
    channel = Channel(phy_model="flora_full", environment="flora", shadowing_std=0.0, use_flora_curves=True)
    phy = FloraPHY(channel)

    snr_values = list(_snr_range(snr_start, snr_stop, snr_step))
    results: list[MonteCarloResult] = []

    for sf in sorted(set(sfs)):
        for payload in sorted(set(payloads)):
            for snr in snr_values:
                logistic = phy.packet_error_rate(snr, sf, payload, per_model="logistic")
                croce = phy.packet_error_rate(snr, sf, payload, per_model="croce")
                logistic_mc = _simulate(logistic, samples, rng)
                croce_mc = _simulate(croce, samples, rng)
                results.append(
                    MonteCarloResult(sf, payload, snr, "logistic", logistic, logistic_mc)
                )
                results.append(
                    MonteCarloResult(sf, payload, snr, "croce", croce, croce_mc)
                )

    header = (
        f"{'SF':>2}  {'Payload':>7}  {'SNR (dB)':>8}  {'Model':>8}  "
        f"{'PER theory':>11}  {'PER MC':>9}  {'Δ':>8}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.sf:>2}  {res.payload:>7}  {res.snr:8.2f}  {res.model:>8}  "
            f"{res.per_theory:11.6f}  {res.per_mc:9.6f}  {res.delta:8.6f}"
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sf", "payload", "snr_dB", "model", "per_theory", "per_mc", "delta"])
            for res in results:
                writer.writerow(
                    [
                        res.sf,
                        res.payload,
                        f"{res.snr:.2f}",
                        res.model,
                        f"{res.per_theory:.6f}",
                        f"{res.per_mc:.6f}",
                        f"{res.delta:.6f}",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, nargs="+", default=[7, 9, 12], help="Spreading factors to evaluate (default: 7 9 12).")
    parser.add_argument(
        "--payload",
        type=int,
        nargs="+",
        default=[13, 51, 127],
        help="Payload sizes in bytes (default: 13 51 127).",
    )
    parser.add_argument("--snr-start", type=float, default=-25.0, help="Minimum SNR in dB (default: -25).")
    parser.add_argument("--snr-stop", type=float, default=5.0, help="Maximum SNR in dB (default: 5).")
    parser.add_argument("--snr-step", type=float, default=5.0, help="Step between SNR points (default: 5).")
    parser.add_argument("--samples", type=int, default=2000, help="Monte-Carlo samples per point (default: 2000).")
    parser.add_argument("--seed", type=int, default=1, help="Random seed controlling the Monte-Carlo draws.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path for the campaign results.")
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.snr_step <= 0:
        raise SystemExit("--snr-step must be positive")

    run_campaign(args.sf, args.payload, args.snr_start, args.snr_stop, args.snr_step, args.samples, args.seed, args.output)


if __name__ == "__main__":
    main()
