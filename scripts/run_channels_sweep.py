"""Run LoRa simulations sweeping the number of channels.

This helper executes ``loraflexsim.run`` for multiple channel
counts and stores the resulting metrics in individual CSV files under
``results``.  The per-channel CSVs are then concatenated into
``results/channels_summary.csv``.

Usage::

    python scripts/run_channels_sweep.py
"""

from __future__ import annotations

import os
import sys
import subprocess

try:  # pandas is optional but required for CSV handling
    import pandas as pd
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"pandas is required for this script: {exc}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    channels_list = [1, 3, 5]
    csv_files: list[str] = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for ch in channels_list:
        out_file = os.path.join(RESULTS_DIR, f"channels_{ch}.csv")
        cmd = [
            sys.executable,
            "-m",
            "loraflexsim.run",
            "--nodes",
            "30",
            "--gateways",
            "1",
            "--steps",
            "500",
            "--mode",
            "random",
            "--interval",
            "10",
            "--seed",
            "1",
            "--runs",
            "5",
            "--channels",
            str(ch),
            "--output",
            out_file,
        ]
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)
        csv_files.append(out_file)

    df = pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)
    summary_path = os.path.join(RESULTS_DIR, "channels_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
