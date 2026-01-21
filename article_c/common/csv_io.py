"""Entrées/sorties CSV."""

from __future__ import annotations

import csv
import logging
import os
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

GROUP_KEYS = ("network_size", "algo", "snir_mode", "seed", "cluster", "mixra_opt_fallback")
EXTRA_MEAN_KEYS = {"mean_toa_s", "mean_latency_s"}

logger = logging.getLogger(__name__)

if os.name == "nt":
    import msvcrt

    @contextmanager
    def _locked_handle(handle) -> object:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield handle
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    @contextmanager
    def _locked_handle(handle) -> object:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Écrit un fichier CSV simple avec verrouillage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        with _locked_handle(handle):
            handle.seek(0, os.SEEK_END)
            write_header = handle.tell() == 0
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(header)
            writer.writerows(rows)


def _coerce_positive_network_size(value: object) -> float:
    if value is None or value == "":
        raise AssertionError("network_size manquant dans les lignes raw.")
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("network_size doit être numérique.") from exc
    if size == 0:
        logger.error("network_size == 0 avant écriture des résultats.")
        assert size != 0, "network_size ne doit pas être égal à 0."
    if size < 0:
        raise ValueError("network_size doit être strictement positif.")
    return size


def aggregate_results(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Agrège les résultats avec moyenne et écart-type par clés de simulation."""
    if "network_size" not in GROUP_KEYS:
        raise AssertionError("network_size doit être inclus dans les clés de regroupement.")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    numeric_keys: set[str] = set()
    for row in raw_rows:
        if row.get("network_size") in (None, "") and row.get("density") not in (None, ""):
            row["network_size"] = row["density"]
        if row.get("network_size") not in (None, ""):
            row["network_size"] = _coerce_positive_network_size(row["network_size"])
        group_key = tuple(row.get(key) for key in GROUP_KEYS)
        groups[group_key].append(row)
        for key, value in row.items():
            if key in GROUP_KEYS or key == "density":
                continue
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    aggregated: list[dict[str, object]] = []
    for group_key, rows in groups.items():
        aggregated_row: dict[str, object] = dict(zip(GROUP_KEYS, group_key))
        if aggregated_row.get("network_size") in (None, ""):
            raise AssertionError("network_size manquant dans les résultats agrégés.")
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


def write_simulation_results(
    output_dir: Path,
    raw_rows: list[dict[str, object]],
    network_size: object | None = None,
) -> None:
    """Écrit les fichiers raw_results.csv et aggregated_results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.csv"
    aggregated_path = output_dir / "aggregated_results.csv"

    expected_network_size = network_size
    if expected_network_size is not None:
        _coerce_positive_network_size(expected_network_size)
    for row in raw_rows:
        row_network_size = row.get("network_size")
        if row_network_size is None or row_network_size == "":
            if row.get("density") not in (None, ""):
                row["network_size"] = row["density"]
            elif expected_network_size == 0.0:
                raise AssertionError(
                    "network_size ne doit pas être remplacé par une valeur par défaut 0.0."
                )
            elif expected_network_size is not None:
                row["network_size"] = expected_network_size
        row_network_size = row.get("network_size")
        _coerce_positive_network_size(row_network_size)

    if raw_rows:
        missing_network_size = [
            row for row in raw_rows if row.get("network_size") in (None, "")
        ]
        if missing_network_size:
            raise AssertionError("network_size manquant dans les lignes raw.")
        network_sizes = sorted({row.get("network_size") for row in raw_rows})
        network_sizes_label = ", ".join(map(str, network_sizes))
        logger.info("network_size written: %s", network_sizes_label)
        print(f"network_size written = {network_sizes_label}")

    raw_header: list[str] = []
    seen: set[str] = set()
    if raw_rows:
        for row in raw_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    raw_header.append(key)
    else:
        raw_header = list(GROUP_KEYS)
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
