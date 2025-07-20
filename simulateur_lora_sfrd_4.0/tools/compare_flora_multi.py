#!/usr/bin/env python3
"""Compare simulator results with multiple FLoRa CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from VERSION_4.launcher.compare_flora import load_flora_metrics
from tools.compare_flora_report import load_run_metrics


def compare_multiple(sim_csv: str | Path, flora_csvs: list[str | Path]) -> pd.DataFrame:
    """Return a DataFrame summarising the differences for each FLoRa CSV."""
    sim_metrics = load_run_metrics(sim_csv)
    rows = []
    for csv_path in flora_csvs:
        flora = load_flora_metrics(csv_path)
        rows.append({
            "file": str(csv_path),
            "PDR_diff": sim_metrics.get("PDR", 0) - flora.get("PDR", 0),
            "collisions_diff": sim_metrics.get("collisions", 0) - flora.get("collisions", 0),
            "throughput_diff": sim_metrics.get("throughput_bps", 0) - flora.get("throughput_bps", 0),
            "energy_diff": sim_metrics.get("energy_J", 0) - flora.get("energy_J", 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare run.py metrics with several FLoRa reference CSV files"
    )
    parser.add_argument("sim_csv", help="CSV produced by VERSION_4/run.py")
    parser.add_argument("flora_csv", nargs="+", help="One or more FLoRa CSV exports")
    parser.add_argument("--output", help="Optional path to save the summary as CSV")
    args = parser.parse_args()

    df = compare_multiple(args.sim_csv, args.flora_csv)
    print(df.to_string(index=False))
    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
