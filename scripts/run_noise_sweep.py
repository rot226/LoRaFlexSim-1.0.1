#!/usr/bin/env python3
"""Run simulations for different noise levels and aggregate results."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

# Allow running the script from a clone without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_noise_level(noise_std: float) -> list[dict[str, str]]:
    """Run the simulator for a given noise standard deviation."""
    output_file = RESULTS_DIR / f"noise_{int(noise_std)}.csv"
    cmd = [
        sys.executable,
        "-m",
        "loraflexsim.run",
        "--nodes",
        "30",
        "--gateways",
        "1",
        "--channels",
        "3",
        "--steps",
        "500",
        "--mode",
        "random",
        "--interval",
        "10",
        "--seed",
        "3",
        "--runs",
        "5",
        "--noise-std",
        str(noise_std),
        "--output",
        str(output_file),
    ]
    subprocess.run(cmd, check=True)
    with output_file.open() as f:
        rows = list(csv.DictReader(f))
        for row in rows:
            row["noise_std"] = str(noise_std)
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    for ns in [0, 1, 2]:
        all_rows.extend(run_noise_level(ns))

    summary_file = RESULTS_DIR / "noise_summary.csv"
    if all_rows:
        fieldnames = ["noise_std"] + [
            "nodes",
            "gateways",
            "channels",
            "mode",
            "interval",
            "steps",
            "run",
            "delivered",
            "collisions",
            "PDR(%)",
            "energy_J",
            "avg_delay",
            "throughput_bps",
        ]
        with summary_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {summary_file}")


if __name__ == "__main__":
    main()
