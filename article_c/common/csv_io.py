"""Entrées/sorties CSV."""

from __future__ import annotations

import csv
import logging
import os
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import math
from statistics import mean, stdev

ROUND_REPLICATION_KEYS = ("round", "replication")
GROUP_KEYS = (
    "network_size",
    "algo",
    "snir_mode",
    "cluster",
    "mixra_opt_fallback",
    *ROUND_REPLICATION_KEYS,
)
BASE_GROUP_KEYS = tuple(
    key for key in GROUP_KEYS if key not in ROUND_REPLICATION_KEYS
)
EXTRA_MEAN_KEYS = {"mean_toa_s", "mean_latency_s"}
EXCLUDED_NUMERIC_KEYS = {"seed", "replication", "node_id", "packet_id"}
DERIVED_SUFFIXES = ("_mean", "_std", "_count", "_ci95", "_p10", "_p50", "_p90")

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


def aggregate_results(
    raw_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Agrège les résultats avec moyenne, écart-type et IC 95% par clés."""
    if "network_size" not in GROUP_KEYS:
        raise AssertionError("network_size doit être inclus dans les clés de regroupement.")
    algo_values = {
        row.get("algo")
        for row in raw_rows
        if row.get("algo") not in (None, "")
    }
    if algo_values:
        for row in raw_rows:
            if row.get("algo") in (None, ""):
                logger.warning(
                    "Algo manquant détecté, séparation stricte appliquée dans l'agrégation."
                )
                row["algo"] = "unknown"
    has_round = any(row.get("round") not in (None, "") for row in raw_rows)
    has_replication = any(row.get("replication") not in (None, "") for row in raw_rows)
    has_intermediate = has_round or has_replication
    if has_intermediate:
        intermediate_rows = _aggregate_rows(
            raw_rows,
            GROUP_KEYS,
            include_base_means=True,
        )
        aggregated_rows = _aggregate_rows(
            intermediate_rows,
            BASE_GROUP_KEYS,
            include_base_means=False,
        )
        return aggregated_rows, intermediate_rows
    aggregated_rows = _aggregate_rows(
        raw_rows,
        BASE_GROUP_KEYS,
        include_base_means=False,
    )
    return aggregated_rows, []


def _aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    *,
    include_base_means: bool,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    numeric_keys = _collect_numeric_keys(rows, group_keys)
    for row in rows:
        if row.get("network_size") in (None, "") and row.get("density") not in (None, ""):
            row["network_size"] = row["density"]
        if row.get("network_size") not in (None, ""):
            row["network_size"] = _coerce_positive_network_size(row["network_size"])
        group_key = tuple(row.get(key) for key in group_keys)
        groups[group_key].append(row)

    aggregated: list[dict[str, object]] = []
    for group_key, grouped_rows in groups.items():
        aggregated_row: dict[str, object] = dict(zip(group_keys, group_key))
        if aggregated_row.get("network_size") in (None, ""):
            raise AssertionError("network_size manquant dans les résultats agrégés.")
        for key in sorted(numeric_keys):
            values = [
                row[key]
                for row in grouped_rows
                if isinstance(row.get(key), (int, float))
            ]
            count = len(values)
            if values:
                mean_value = mean(values)
                std_value = stdev(values) if count > 1 else 0.0
            else:
                mean_value = 0.0
                std_value = 0.0
            ci95_value = 1.96 * std_value / math.sqrt(count) if count > 1 else 0.0
            aggregated_row[f"{key}_mean"] = mean_value
            aggregated_row[f"{key}_std"] = std_value
            aggregated_row[f"{key}_count"] = count
            aggregated_row[f"{key}_ci95"] = ci95_value
            sorted_values = sorted(values)
            aggregated_row[f"{key}_p10"] = _percentile(sorted_values, 10)
            aggregated_row[f"{key}_p50"] = _percentile(sorted_values, 50)
            aggregated_row[f"{key}_p90"] = _percentile(sorted_values, 90)
            if include_base_means or key in EXTRA_MEAN_KEYS:
                aggregated_row[key] = mean_value
        aggregated.append(aggregated_row)
    return aggregated


def _collect_numeric_keys(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
) -> set[str]:
    numeric_keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in group_keys or key == "density" or key in EXCLUDED_NUMERIC_KEYS:
                continue
            if any(key.endswith(suffix) for suffix in DERIVED_SUFFIXES):
                continue
            if isinstance(value, (int, float)):
                numeric_keys.add(key)
    return numeric_keys


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower]) + (float(values[upper]) - float(values[lower])) * weight


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

    aggregated_rows, intermediate_rows = aggregate_results(raw_rows)
    aggregated_header = (
        list(aggregated_rows[0].keys()) if aggregated_rows else list(BASE_GROUP_KEYS)
    )
    write_rows(
        aggregated_path,
        aggregated_header,
        [[row.get(key, "") for key in aggregated_header] for row in aggregated_rows],
    )
    if intermediate_rows:
        has_round = any(row.get("round") not in (None, "") for row in intermediate_rows)
        if has_round:
            intermediate_name = "aggregated_results_by_round.csv"
        else:
            intermediate_name = "aggregated_results_by_replication.csv"
        intermediate_path = output_dir / intermediate_name
        intermediate_header = list(intermediate_rows[0].keys())
        write_rows(
            intermediate_path,
            intermediate_header,
            [
                [row.get(key, "") for key in intermediate_header]
                for row in intermediate_rows
            ],
        )
