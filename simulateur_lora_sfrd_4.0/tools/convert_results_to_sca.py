#!/usr/bin/env python3
"""Convert simulator CSV results to OMNeT++ .sca files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_sca(row: pd.Series, path: Path) -> None:
    """Write one run of metrics to ``path`` in .sca format."""
    run_prefix = f"run{int(row.get('run', 1))}"
    with open(path, "w") as f:
        def scalar(name: str, value: float) -> None:
            f.write(f"scalar {run_prefix} {name} {value}\n")

        delivered = float(row.get("delivered", 0))
        collisions = float(row.get("collisions", 0))
        sent = delivered + collisions
        scalar("sent", sent)
        scalar("received", delivered)
        scalar("collisions", collisions)
        if "throughput_bps" in row:
            scalar("throughput_bps", float(row["throughput_bps"]))
        if "energy" in row:
            scalar("energy_J", float(row["energy"]))
        if "avg_delay" in row:
            scalar("avg_delay_s", float(row["avg_delay"]))
        for sf in range(7, 13):
            key = f"sf{sf}"
            if key in row:
                scalar(key, float(row[key]))
            coll_key = f"collisions_sf{sf}"
            if coll_key in row:
                scalar(coll_key, float(row[coll_key]))


def convert(csv_file: str | Path, output_dir: str | Path) -> None:
    df = pd.read_csv(csv_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        run = int(row.get("run", 1))
        path = out / f"run{run}.sca"
        write_sca(row, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV results to .sca")
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("output", help="Output directory for .sca files")
    args = parser.parse_args()
    convert(args.csv, args.output)


if __name__ == "__main__":
    main()
