"""Run a sweep over different traffic intervals.

This utility executes the base simulator for a set of packet interval values
and collects the per-run metrics produced by ``loraflexsim/run.py``.
Results for each interval are stored in ``results/interval_<IV>.csv`` and all
rows are aggregated into ``results/interval_summary.csv``.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

# Paths relative to repository root
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    intervals = [20, 10, 5]
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_paths: list[Path] = []

    for iv in intervals:
        out_file = RESULTS_DIR / f"interval_{iv}.csv"
        cmd = [
            "python",
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
            "--seed",
            "2",
            "--runs",
            "5",
            "--interval",
            str(iv),
            "--output",
            str(out_file),
        ]
        subprocess.run(cmd, check=True)
        csv_paths.append(out_file)

    # Merge the individual CSV files into a single summary
    summary_path = RESULTS_DIR / "interval_summary.csv"
    header: list[str] | None = None
    rows: list[list[str]] = []
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            rows.extend(list(reader))

    if header is None:
        raise RuntimeError("No data collected")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {summary_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
