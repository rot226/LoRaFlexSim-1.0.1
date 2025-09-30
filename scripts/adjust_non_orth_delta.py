#!/usr/bin/env python3
"""Utility to build a NON_ORTH_DELTA table from measurements.

The expected input is a CSV file containing at least the columns
``sf_signal``, ``sf_interference`` and ``delta_dB``.  Each row represents the
measured minimum power difference (in dB) for a packet of spreading factor
``sf_signal`` to capture against an interferer of spreading factor
``sf_interference``.  Multiple rows for the same pair are averaged.

The resulting 6x6 matrix is written to a JSON file that can be loaded by the
simulator through the ``load_non_orth_delta`` helper.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV file with measurements or OMNeT++ traces")
    parser.add_argument(
        "-o",
        "--output",
        default="non_orth_delta.json",
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    matrix: list[list[list[float] | None]] = [[[] for _ in range(6)] for _ in range(6)]

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sf_s = int(row["sf_signal"])
                sf_i = int(row["sf_interference"])
                delta = float(row["delta_dB"])
            except (KeyError, ValueError):
                continue
            i, j = sf_s - 7, sf_i - 7
            if 0 <= i < 6 and 0 <= j < 6:
                matrix[i][j].append(delta)

    result: list[list[float]] = [[0.0] * 6 for _ in range(6)]
    for i in range(6):
        for j in range(6):
            samples = matrix[i][j]
            if samples:
                result[i][j] = statistics.mean(samples)
            else:
                result[i][j] = DEFAULT_NON_ORTH_DELTA[i][j]

    with open(args.output, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    print(f"Matrix written to {args.output}")


if __name__ == "__main__":
    main()
