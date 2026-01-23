"""Entrées/sorties CSV."""

from __future__ import annotations

import csv
import logging
import os
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import math
from statistics import mean, median, stdev

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
EXCLUDED_NUMERIC_KEYS = {"seed", "replication", "round", "node_id", "packet_id"}
SUM_KEYS = {"success", "failure"}
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


def _parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _normalize_snir_mode(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"snir_on", "on", "true", "1", "yes"}:
        return "snir_on"
    if text in {"snir_off", "off", "false", "0", "no"}:
        return "snir_off"
    if text in {"snir_unknown", "unknown", "n/a", "na"}:
        return "snir_unknown"
    return None


def _normalize_group_keys(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row.get("algo") in (None, "") and row.get("algorithm") not in (None, ""):
            row["algo"] = row.get("algorithm")
        if row.get("snir_mode") in (None, ""):
            snir_mode = _normalize_snir_mode(row.get("snir_state") or row.get("snir"))
            if snir_mode is None:
                snir_flag = _parse_bool(row.get("with_snir"))
                if snir_flag is None:
                    snir_flag = _parse_bool(row.get("use_snir"))
                if snir_flag is True:
                    snir_mode = "snir_on"
                elif snir_flag is False:
                    snir_mode = "snir_off"
            if snir_mode is not None:
                row["snir_mode"] = snir_mode


def _log_control_table(rows: list[dict[str, object]], label: str) -> None:
    if not rows:
        return
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        algo = str(row.get("algo") or row.get("algorithm") or "unknown")
        snir_mode = (
            _normalize_snir_mode(row.get("snir_mode"))
            or _normalize_snir_mode(row.get("snir_state"))
            or _normalize_snir_mode(row.get("snir"))
        )
        if snir_mode is None:
            snir_flag = _parse_bool(row.get("with_snir"))
            if snir_flag is None:
                snir_flag = _parse_bool(row.get("use_snir"))
            if snir_flag is True:
                snir_mode = "snir_on"
            elif snir_flag is False:
                snir_mode = "snir_off"
            else:
                snir_mode = "snir_unknown"
        size_value = row.get("network_size") or row.get("density")
        size_label = "unknown"
        if size_value not in (None, ""):
            try:
                size_label = str(int(round(float(size_value))))
            except (TypeError, ValueError):
                size_label = str(size_value)
        counts[(algo, snir_mode, size_label)] += 1
    print(f"Tableau de contrôle ({label}):")
    print("algo\tsnir_mode\tnetwork_size\tcount")
    for (algo, snir_mode, size_label), count in sorted(counts.items()):
        print(f"{algo}\t{snir_mode}\t{size_label}\t{count}")


def _log_reward_min_max(raw_rows: list[dict[str, object]]) -> None:
    if not raw_rows:
        return
    has_reward_key = any("reward" in row for row in raw_rows)
    if not has_reward_key:
        logger.info("Colonne reward absente des lignes raw; skip diagnostic.")
        return
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    invalid_values: list[object] = []
    missing_count = 0
    for row in raw_rows:
        if "reward" not in row:
            continue
        reward = row.get("reward")
        if reward in (None, ""):
            missing_count += 1
            continue
        try:
            value = float(reward)
        except (TypeError, ValueError):
            invalid_values.append(reward)
            continue
        if not math.isfinite(value):
            raise AssertionError(f"reward non fini détecté: {reward}")
        algo = str(row.get("algo") or row.get("algorithm") or "unknown")
        size_value = row.get("network_size") or row.get("density")
        if size_value in (None, ""):
            size_label = "unknown"
        else:
            try:
                size_label = str(int(round(float(size_value))))
            except (TypeError, ValueError):
                size_label = str(size_value)
        groups[(algo, size_label)].append(value)
    if invalid_values:
        logger.warning("Valeurs reward non numériques ignorées: %s", invalid_values[:5])
    if missing_count:
        logger.warning("Valeurs reward manquantes détectées: %s", missing_count)
    if not groups:
        raise AssertionError("Aucune valeur reward numérique disponible avant agrégation.")
    for (algo, size_label), values in sorted(groups.items()):
        if not values:
            logger.warning("Valeurs reward vides pour %s/%s.", algo, size_label)
            continue
        min_value = min(values)
        max_value = max(values)
        logger.info(
            "Reward min/max avant agrégation [%s - %s]: %.6f/%.6f",
            algo,
            size_label,
            min_value,
            max_value,
        )
        print(
            "Reward min/max avant agrégation "
            f"[{algo} - {size_label}]: {min_value:.6f}/{max_value:.6f}"
        )


def _log_metric_summary(
    raw_rows: list[dict[str, object]],
    metrics: tuple[str, ...],
) -> None:
    if not raw_rows:
        return
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in raw_rows:
        algo = str(row.get("algo") or row.get("algorithm") or "unknown")
        size_value = row.get("network_size") or row.get("density")
        size_label = "unknown"
        if size_value not in (None, ""):
            try:
                size_label = str(int(round(float(size_value))))
            except (TypeError, ValueError):
                size_label = str(size_value)
        for metric in metrics:
            if metric not in row:
                continue
            value = row.get(metric)
            if value in (None, ""):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            groups[(algo, size_label)][metric].append(numeric_value)
    if not groups:
        return
    print("Statistiques brutes (min/max/median) par algo et taille:")
    print("algo\tnetwork_size\tmetric\tmin\tmax\tmedian\tcount")
    for (algo, size_label), metric_map in sorted(groups.items()):
        for metric in metrics:
            values = metric_map.get(metric, [])
            if not values:
                continue
            print(
                f"{algo}\t{size_label}\t{metric}\t"
                f"{min(values):.6f}\t{max(values):.6f}\t"
                f"{median(values):.6f}\t{len(values)}"
            )


def aggregate_results(
    raw_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Agrège les résultats avec moyenne, écart-type et IC 95% par clés."""
    if "network_size" not in GROUP_KEYS:
        raise AssertionError("network_size doit être inclus dans les clés de regroupement.")
    _normalize_group_keys(raw_rows)
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
            sum_value = sum(values)
            if values:
                mean_value = sum_value / count
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
            if key in SUM_KEYS:
                aggregated_row[key] = sum_value
            elif include_base_means or key in EXTRA_MEAN_KEYS:
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
        _normalize_group_keys(raw_rows)
        missing_network_size = [
            row for row in raw_rows if row.get("network_size") in (None, "")
        ]
        if missing_network_size:
            raise AssertionError("network_size manquant dans les lignes raw.")
        network_sizes = sorted({row.get("network_size") for row in raw_rows})
        network_sizes_label = ", ".join(map(str, network_sizes))
        logger.info("network_size written: %s", network_sizes_label)
        print(f"network_size written = {network_sizes_label}")
        _log_reward_min_max(raw_rows)
        _log_metric_summary(
            raw_rows,
            (
                "reward",
                "success_rate",
                "throughput_success",
                "energy_per_success",
            ),
        )

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

    _log_control_table(raw_rows, "raw_results.csv")
    aggregated_rows, intermediate_rows = aggregate_results(raw_rows)
    _log_control_table(aggregated_rows, "aggregated_results.csv")
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
