#!/usr/bin/env python3
"""Convert FLoRa OMNeT++ results to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from VERSION_4.launcher.compare_flora import _parse_sca_file


def gather_sca_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.sca")))
        elif path.suffix.lower() == ".sca":
            files.append(path)
    return files


def convert(paths: list[str | Path], output: str | Path) -> None:
    files = gather_sca_files(paths)
    rows = []
    for idx, sca in enumerate(files, start=1):
        row = _parse_sca_file(sca)
        row["run"] = idx
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FLoRa .sca/.vec to CSV")
    parser.add_argument("results", nargs="+", help=".sca files or directories")
    parser.add_argument("--output", "-o", default="flora.csv", help="Output CSV file")
    args = parser.parse_args()
    convert(args.results, args.output)


if __name__ == "__main__":
    main()
