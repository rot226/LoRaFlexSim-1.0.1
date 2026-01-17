"""Entrées/sorties CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

GROUP_KEYS = ("density", "algo", "snir_mode", "cluster")
EXTRA_MEAN_KEYS = {"mean_toa_s", "mean_latency_s"}


def write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Écrit un fichier CSV simple."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def aggregate_results(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Agrège les résultats avec moyenne et écart-type par (density, algo, snir_mode)."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    numeric_keys: set[str] = set()
    for row in raw_rows:
        group_key = tuple(row.get(key) for key in GROUP_KEYS)
        groups[group_key].append(row)
        for key, value in row.items():
            if key in GROUP_KEYS:
                continue
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    aggregated: list[dict[str, object]] = []
    for group_key, rows in groups.items():
        aggregated_row: dict[str, object] = dict(zip(GROUP_KEYS, group_key))
        for key in sorted(numeric_keys):
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                aggregated_row[f"{key}_mean"] = mean(values)
                aggregated_row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
            else:
                aggregated_row[f"{key}_mean"] = 0.0
                aggregated_row[f"{key}_std"] = 0.0
            if key in EXTRA_MEAN_KEYS:
                aggregated_row[key] = aggregated_row[f"{key}_mean"]
        aggregated.append(aggregated_row)
    return aggregated


def write_simulation_results(output_dir: Path, raw_rows: list[dict[str, object]]) -> None:
    """Écrit les fichiers raw_results.csv et aggregated_results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.csv"
    aggregated_path = output_dir / "aggregated_results.csv"

    raw_header = list(raw_rows[0].keys()) if raw_rows else list(GROUP_KEYS)
    write_rows(
        raw_path,
        raw_header,
        [[row.get(key, "") for key in raw_header] for row in raw_rows],
    )

    aggregated_rows = aggregate_results(raw_rows)
    aggregated_header = list(aggregated_rows[0].keys()) if aggregated_rows else list(GROUP_KEYS)
    write_rows(
        aggregated_path,
        aggregated_header,
        [[row.get(key, "") for key in aggregated_header] for row in aggregated_rows],
    )
