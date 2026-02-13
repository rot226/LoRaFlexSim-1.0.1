"""Entrées/sorties CSV."""

from __future__ import annotations

import csv
import logging
import os
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import math

from article_c.common.config import normalize_algorithm, normalize_cluster, normalize_snir_mode
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
STEP1_EXPECTED_METRICS = ("sent", "received", "pdr")
STEP2_EXPECTED_METRICS = ("reward", "success_rate")
MERGE_KEYS = ("network_size", "algo", "snir_mode", "cluster", "replication")
CANONICAL_ID_COLUMNS = (
    "network_size",
    "algo",
    "snir_mode",
    "cluster",
    "mixra_opt_fallback",
)

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


def atomic_write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Écrit atomiquement un CSV via `path.tmp` puis `os.replace`.

    Un verrou de fichier dédié est conservé pour sérialiser les écritures
    concurrentes vers le même fichier de destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    tmp_path = path.with_name(f"{path.name}.tmp")
    with lock_path.open("a+", newline="", encoding="utf-8") as lock_handle:
        with _locked_handle(lock_handle):
            try:
                with tmp_path.open("w", newline="", encoding="utf-8") as tmp_handle:
                    writer = csv.writer(tmp_handle)
                    writer.writerow(header)
                    writer.writerows(rows)
                    tmp_handle.flush()
                    os.fsync(tmp_handle.fileno())
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()


def append_or_merge_csv(
    path: Path,
    header: list[str],
    rows: list[list[object]],
    dedupe_keys: tuple[str, ...],
    canonical_base_columns: tuple[str, ...] | None = None,
) -> None:
    """Fusionne un CSV avec l'existant, puis réécrit atomiquement.

    Étapes:
    1. Lecture de l'existant si présent.
    2. Normalisation du schéma (union des colonnes).
    3. Concaténation + déduplication sur `dedupe_keys`.
    4. Réécriture atomique du CSV fusionné.

    Le verrouillage de fichier est conservé via un lock dédié `<path>.lock`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    tmp_path = path.with_name(f"{path.name}.tmp")

    incoming_rows = [
        {column: row[index] if index < len(row) else "" for index, column in enumerate(header)}
        for row in rows
    ]

    with lock_path.open("a+", newline="", encoding="utf-8") as lock_handle:
        with _locked_handle(lock_handle):
            existing_header: list[str] = []
            existing_rows: list[dict[str, object]] = []
            if path.exists():
                with path.open("r", newline="", encoding="utf-8") as existing_handle:
                    reader = csv.DictReader(existing_handle)
                    if reader.fieldnames:
                        existing_header = list(reader.fieldnames)
                    for row in reader:
                        existing_rows.append(dict(row))

            merged_header = _merge_headers(
                existing_header,
                header,
                canonical_base_columns=canonical_base_columns,
            )
            if not merged_header:
                merged_header = list(header)

            for row in existing_rows:
                for column in merged_header:
                    row.setdefault(column, _missing_value_for_column(column))
            for row in incoming_rows:
                for column in merged_header:
                    row.setdefault(column, _missing_value_for_column(column))

            deduped_rows_by_key: dict[tuple[object, ...], dict[str, object]] = {}
            for row in [*existing_rows, *incoming_rows]:
                row_key = tuple(str(row.get(key, "")).strip() for key in dedupe_keys)
                deduped_rows_by_key[row_key] = row

            merged_rows = [
                [row.get(column, "") for column in merged_header]
                for row in deduped_rows_by_key.values()
            ]

            try:
                with tmp_path.open("w", newline="", encoding="utf-8") as tmp_handle:
                    writer = csv.writer(tmp_handle)
                    writer.writerow(merged_header)
                    writer.writerows(merged_rows)
                    tmp_handle.flush()
                    os.fsync(tmp_handle.fileno())
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()


def _coerce_positive_network_size(value: object) -> int:
    if value is None or value == "":
        raise AssertionError("network_size manquant dans les lignes raw.")
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("network_size doit être numérique.") from exc
    if not math.isfinite(size):
        raise ValueError("network_size doit être une valeur finie.")
    if size == 0:
        logger.error("network_size == 0 avant écriture des résultats.")
        assert size != 0, "network_size ne doit pas être égal à 0."
    if size < 0:
        raise ValueError("network_size doit être strictement positif.")
    if not size.is_integer():
        raise ValueError("network_size doit être un entier.")
    return int(size)


def resolve_step2_input_csv_paths(results_dir: Path) -> list[Path]:
    """Résout les CSV d'entrée Step2 selon l'ordre de priorité contractuel.

    Ordre:
    1. `results/by_size/size_*/rep_*/raw_results.csv`
    2. `results/aggregates/aggregated_results.csv`
    3. `FileNotFoundError` explicite.
    """
    nested_raw_paths = sorted(results_dir.glob("by_size/size_*/rep_*/raw_results.csv"))
    if nested_raw_paths:
        return nested_raw_paths

    aggregated_path = results_dir / "aggregates" / "aggregated_results.csv"
    if aggregated_path.exists():
        return [aggregated_path]

    nested_pattern = results_dir / "by_size" / "size_*" / "rep_*" / "raw_results.csv"
    raise FileNotFoundError(
        "Aucun CSV Step2 trouvé. Chemins recherchés (dans cet ordre): "
        f"{nested_pattern} puis {aggregated_path}."
    )


def _missing_value_for_column(column: str) -> object:
    if column in {"network_size", "density"}:
        return math.nan
    if any(column.endswith(suffix) for suffix in DERIVED_SUFFIXES):
        return math.nan
    return ""


def _merge_headers(
    existing_header: list[str],
    incoming_header: list[str],
    *,
    canonical_base_columns: tuple[str, ...] | None,
) -> list[str]:
    merged_header: list[str] = []
    for candidate_header in (existing_header, incoming_header):
        for column in candidate_header:
            if column not in merged_header:
                merged_header.append(column)
    if not canonical_base_columns:
        return merged_header
    ordered_header = [
        column for column in canonical_base_columns if column in merged_header
    ]
    ordered_header.extend(column for column in merged_header if column not in ordered_header)
    return ordered_header


def _canonical_columns_for_step(
    expected_metrics: tuple[str, ...],
    *,
    include_round: bool,
    include_replication: bool,
) -> tuple[str, ...]:
    metric_columns: list[str] = []
    for metric in expected_metrics:
        metric_columns.extend(
            [
                metric,
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_count",
                f"{metric}_ci95",
                f"{metric}_p10",
                f"{metric}_p50",
                f"{metric}_p90",
            ]
        )
    rep_columns: list[str] = []
    if include_round:
        rep_columns.append("round")
    if include_replication:
        rep_columns.append("replication")
    return (*CANONICAL_ID_COLUMNS, *rep_columns, *metric_columns)


def _coerce_density(value: object) -> float:
    try:
        density = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("density doit être numérique.") from exc
    if not math.isfinite(density):
        raise ValueError("density doit être une valeur finie.")
    if density < 0:
        raise ValueError("density doit être positive.")
    return float(density)


def _resolve_network_size_with_density_fallback(row: dict[str, object]) -> int:
    network_size = row.get("network_size")
    if network_size in (None, "") and row.get("density") not in (None, ""):
        network_size = row.get("density")
    return _coerce_positive_network_size(network_size)


def _parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_computed_density(rows: list[dict[str, object]]) -> bool:
    for row in rows:
        density_value = _parse_float(row.get("density"))
        if density_value is None:
            continue
        if not math.isfinite(density_value):
            continue
        network_size_value = _parse_float(row.get("network_size"))
        if network_size_value is None:
            return True
        if not math.isfinite(network_size_value):
            continue
        if not math.isclose(
            float(density_value),
            float(network_size_value),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return True
    return False


def _normalize_snir_mode(value: object) -> str | None:
    return normalize_snir_mode(value, default=None)


def _normalize_cluster(value: object) -> str:
    return normalize_cluster(value, default="all")


def _normalize_algo(value: object) -> str:
    normalized = normalize_algorithm(value, default=None)
    if normalized is not None:
        return normalized
    text = str(value).strip() if value is not None else ""
    return text


def _is_failed_status(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() == "failed"


def _normalize_group_keys(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row.get("algo") in (None, "") and row.get("algorithm") not in (None, ""):
            row["algo"] = row.get("algorithm")
        row["algo"] = _normalize_algo(row.get("algo"))
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
        elif (normalized_snir := _normalize_snir_mode(row.get("snir_mode"))) is not None:
            row["snir_mode"] = normalized_snir
        row["cluster"] = _normalize_cluster(row.get("cluster"))


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


def _log_replication_status_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    rep_status: dict[tuple[str, str, str], dict[object, bool]] = defaultdict(dict)
    for index, row in enumerate(rows):
        algo = str(row.get("algo") or row.get("algorithm") or "unknown")
        snir_mode = (
            _normalize_snir_mode(row.get("snir_mode"))
            or _normalize_snir_mode(row.get("snir_state"))
            or _normalize_snir_mode(row.get("snir"))
            or "snir_unknown"
        )
        size_value = row.get("network_size") or row.get("density")
        size_label = "unknown"
        if size_value not in (None, ""):
            try:
                size_label = str(int(round(float(size_value))))
            except (TypeError, ValueError):
                size_label = str(size_value)

        rep_id_parts = tuple(
            row.get(key) for key in ROUND_REPLICATION_KEYS if row.get(key) not in (None, "")
        )
        rep_id: object = rep_id_parts if rep_id_parts else ("row", index)
        group_key = (size_label, algo, snir_mode)
        has_failed = rep_status[group_key].get(rep_id, False)
        rep_status[group_key][rep_id] = has_failed or _is_failed_status(row.get("status"))

    print("Résumé réplications valides/échouées par size/algo/snir:")
    print("network_size\talgo\tsnir_mode\tvalid_reps\tfailed_reps")
    for (size_label, algo, snir_mode), status_map in sorted(rep_status.items()):
        failed_count = sum(1 for is_failed in status_map.values() if is_failed)
        valid_count = len(status_map) - failed_count
        print(f"{size_label}\t{algo}\t{snir_mode}\t{valid_count}\t{failed_count}")


def aggregate_results(
    raw_rows: list[dict[str, object]],
    *,
    expected_metrics: tuple[str, ...],
    step_label: str,
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
            expected_metrics=expected_metrics,
            step_label=step_label,
        )
        aggregated_rows = _aggregate_rows(
            intermediate_rows,
            BASE_GROUP_KEYS,
            include_base_means=False,
            expected_metrics=expected_metrics,
            step_label=step_label,
        )
        if "received" in expected_metrics:
            _rewrite_received_mean(intermediate_rows)
            _rewrite_received_mean(aggregated_rows)
        return aggregated_rows, intermediate_rows
    aggregated_rows = _aggregate_rows(
        raw_rows,
        BASE_GROUP_KEYS,
        include_base_means=False,
        expected_metrics=expected_metrics,
        step_label=step_label,
    )
    if "received" in expected_metrics:
        _rewrite_received_mean(aggregated_rows)
    return aggregated_rows, []


def _aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    *,
    include_base_means: bool,
    expected_metrics: tuple[str, ...],
    step_label: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    missing_expected_metrics = [
        metric for metric in expected_metrics if not any(metric in row for row in rows)
    ]
    for metric in missing_expected_metrics:
        logger.warning(
            "[%s] Métrique attendue absente des lignes raw: %s. "
            "Les percentiles associés ne seront pas calculés.",
            step_label,
            metric,
        )
    numeric_keys = _collect_numeric_keys(rows, group_keys) - set(missing_expected_metrics)
    for row in rows:
        row["network_size"] = _resolve_network_size_with_density_fallback(row)
        group_key = tuple(row.get(key) for key in group_keys)
        groups[group_key].append(row)

    aggregated: list[dict[str, object]] = []
    for group_key, grouped_rows in groups.items():
        aggregated_row: dict[str, object] = dict(zip(group_keys, group_key))
        valid_grouped_rows = [
            row for row in grouped_rows if not _is_failed_status(row.get("status"))
        ]
        if not valid_grouped_rows:
            logger.warning(
                "[%s] Groupe sans réplication valide (toutes status=failed): %s",
                step_label,
                {
                    key: value
                    for key, value in zip(group_keys, group_key)
                    if key in {"network_size", "algo", "snir_mode"}
                },
            )
        if aggregated_row.get("network_size") in (None, ""):
            raise AssertionError("network_size manquant dans les résultats agrégés.")
        aggregated_row["network_size"] = _coerce_positive_network_size(
            aggregated_row["network_size"]
        )
        aggregated_row.pop("density", None)
        for key in sorted(numeric_keys):
            values = [
                row[key]
                for row in valid_grouped_rows
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


def _rewrite_received_mean(rows: list[dict[str, object]]) -> None:
    for row in rows:
        sent = row.get("sent_mean")
        pdr = row.get("pdr_mean")
        if isinstance(sent, (int, float)) and isinstance(pdr, (int, float)):
            row["received_mean"] = sent * pdr


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


def _extract_size_rep_from_path(path: Path) -> tuple[int, int] | None:
    size_value: int | None = None
    rep_value: int | None = None
    for part in path.parts:
        if part.startswith("size_"):
            suffix = part[len("size_") :]
            if suffix.isdigit():
                size_value = int(suffix)
        elif part.startswith("rep_"):
            suffix = part[len("rep_") :]
            if suffix.isdigit():
                rep_value = int(suffix)
    if size_value is None or rep_value is None:
        return None
    return size_value, rep_value


def aggregate_results_by_size(
    results_dir: Path,
    *,
    write_global_aggregated: bool = False,
    by_size_dirname: str = "by_size",
) -> dict[str, int]:
    """Consolide les `aggregated_results.csv` des réplications par taille.

    Écrit, pour chaque dossier `size_<N>`:
    - `results/by_size/size_<N>/aggregated_results.csv`

    Et, si demandé, un fichier global:
    - `results/aggregates/aggregated_results.csv`
    """

    by_size_dir = results_dir / by_size_dirname
    size_dirs = [path for path in sorted(by_size_dir.glob("size_*")) if path.is_dir()]
    if not size_dirs:
        return {
            "size_count": 0,
            "size_row_count": 0,
            "global_row_count": 0,
        }

    all_rows_for_global: list[dict[str, str]] = []
    total_size_rows = 0
    size_count = 0

    for size_dir in size_dirs:
        rep_paths = sorted(size_dir.glob("rep_*/aggregated_results.csv"))
        if not rep_paths:
            continue

        size_rows: list[dict[str, str]] = []
        header_order: list[str] = []
        seen_headers: set[str] = set()
        for rep_path in rep_paths:
            with rep_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = [name for name in (reader.fieldnames or []) if name is not None]
                for fieldname in fieldnames:
                    if fieldname not in seen_headers:
                        seen_headers.add(fieldname)
                        header_order.append(fieldname)
                for row in reader:
                    if not row or all(value in (None, "") for value in row.values()):
                        continue
                    normalized_row = {
                        key: "" if value is None else str(value)
                        for key, value in row.items()
                        if key is not None
                    }
                    normalized_row["network_size"] = str(
                        _resolve_network_size_with_density_fallback(normalized_row)
                    )
                    normalized_row.pop("density", None)
                    normalized_row["source_size_dir"] = size_dir.name
                    size_rows.append(normalized_row)

        if not size_rows:
            continue

        if "source_size_dir" not in seen_headers:
            header_order.append("source_size_dir")
        if "network_size" not in seen_headers:
            header_order.insert(0, "network_size")
        if "density" in header_order:
            header_order.remove("density")

        atomic_write_csv(
            size_dir / "aggregated_results.csv",
            header_order,
            [[row.get(column, "") for column in header_order] for row in size_rows],
        )
        size_count += 1
        total_size_rows += len(size_rows)
        all_rows_for_global.extend(size_rows)

    global_count = 0
    if write_global_aggregated and all_rows_for_global:
        global_header: list[str] = []
        seen_columns: set[str] = set()
        for row in all_rows_for_global:
            for column in row.keys():
                if column not in seen_columns:
                    seen_columns.add(column)
                    global_header.append(column)
        atomic_write_csv(
            results_dir / "aggregates" / "aggregated_results.csv",
            global_header,
            [
                [row.get(column, "") for column in global_header]
                for row in all_rows_for_global
            ],
        )
        global_count = len(all_rows_for_global)

    return {
        "size_count": size_count,
        "size_row_count": total_size_rows,
        "global_row_count": global_count,
    }


def _find_run_status_path(output_dir: Path, step_label: str) -> Path:
    filename = f"run_status_{step_label.lower()}.csv"
    for candidate in [output_dir, *output_dir.parents]:
        candidate_path = candidate / filename
        if candidate_path.exists():
            return candidate_path
    for parent in output_dir.parents:
        if parent.name == "by_size":
            return parent.parent / filename
    return output_dir / filename


def _log_network_size_mismatch(
    output_dir: Path,
    step_label: str,
    expected_size: int,
    rep_value: int,
    *,
    detail: str,
) -> None:
    status_path = _find_run_status_path(output_dir, step_label)
    write_rows(
        status_path,
        [
            "status",
            "step",
            "network_size",
            "replication",
            "seed",
            "algorithm",
            "snir_mode",
            "error",
        ],
        [
            [
                "failed",
                step_label.lower(),
                expected_size,
                rep_value,
                "",
                "",
                "",
                detail,
            ]
        ],
    )


def _validate_replication_network_size(
    output_dir: Path,
    raw_rows: list[dict[str, object]],
    *,
    step_label: str,
) -> None:
    size_rep = _extract_size_rep_from_path(output_dir)
    if size_rep is None:
        return
    expected_size, rep_value = size_rep
    observed_sizes = {
        _coerce_positive_network_size(row.get("network_size")) for row in raw_rows
    }
    invalid_sizes = sorted(size for size in observed_sizes if size != expected_size)
    if not invalid_sizes:
        return
    detail = (
        "Incohérence network_size pour écriture sous "
        f"size_{expected_size}/rep_{rep_value}: valeurs trouvées {invalid_sizes}."
    )
    _log_network_size_mismatch(
        output_dir,
        step_label,
        expected_size,
        rep_value,
        detail=detail,
    )
    raise ValueError(detail)


def write_simulation_results(
    output_dir: Path,
    raw_rows: list[dict[str, object]],
    network_size: object | None = None,
) -> None:
    """Écrit raw_results.csv, raw_all.csv, raw_cluster.csv et aggregated_results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_replication_network_size(output_dir, raw_rows, step_label="step2")
    raw_path = output_dir / "raw_results.csv"
    raw_all_path = output_dir / "raw_all.csv"
    raw_cluster_path = output_dir / "raw_cluster.csv"
    aggregated_path = output_dir / "aggregated_results.csv"

    expected_network_size = network_size
    if expected_network_size is not None:
        _coerce_positive_network_size(expected_network_size)
    has_computed_density = _has_computed_density(raw_rows)
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
        row["network_size"] = _coerce_positive_network_size(row_network_size)
        if row.get("density") in (None, "") and not has_computed_density:
            row["density"] = _coerce_density(row["network_size"])
        elif row.get("density") not in (None, ""):
            row["density"] = _coerce_density(row["density"])
        row["cluster"] = _normalize_cluster(row.get("cluster"))

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
    _log_replication_status_summary(raw_rows)

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
    atomic_write_csv(
        raw_path,
        raw_header,
        [[row.get(key, "") for key in raw_header] for row in raw_rows],
    )
    raw_all_rows = [row for row in raw_rows if row.get("cluster") == "all"]
    raw_cluster_rows = [row for row in raw_rows if row.get("cluster") != "all"]
    atomic_write_csv(
        raw_all_path,
        raw_header,
        [[row.get(key, "") for key in raw_header] for row in raw_all_rows],
    )
    atomic_write_csv(
        raw_cluster_path,
        raw_header,
        [[row.get(key, "") for key in raw_header] for row in raw_cluster_rows],
    )

    _log_control_table(raw_rows, "raw_results.csv")
    aggregated_rows, intermediate_rows = aggregate_results(
        raw_rows,
        expected_metrics=STEP2_EXPECTED_METRICS,
        step_label="Step2",
    )
    _log_control_table(aggregated_rows, "aggregated_results.csv")
    for row in aggregated_rows:
        row["network_size"] = _resolve_network_size_with_density_fallback(row)
        row.pop("density", None)
    aggregated_header = (
        list(aggregated_rows[0].keys()) if aggregated_rows else list(BASE_GROUP_KEYS)
    )
    if "density" in aggregated_header:
        aggregated_header.remove("density")
    append_or_merge_csv(
        aggregated_path,
        aggregated_header,
        [[row.get(key, "") for key in aggregated_header] for row in aggregated_rows],
        dedupe_keys=_step2_dedupe_keys(aggregated_rows),
        canonical_base_columns=_canonical_columns_for_step(
            STEP2_EXPECTED_METRICS,
            include_round=False,
            include_replication=False,
        ),
    )
    if intermediate_rows:
        has_round = any(row.get("round") not in (None, "") for row in intermediate_rows)
        if has_round:
            intermediate_name = "aggregated_results_by_round.csv"
        else:
            intermediate_name = "aggregated_results_by_replication.csv"
        intermediate_path = output_dir / intermediate_name
        intermediate_header = list(intermediate_rows[0].keys())
        append_or_merge_csv(
            intermediate_path,
            intermediate_header,
            [
                [row.get(key, "") for key in intermediate_header]
                for row in intermediate_rows
            ],
            dedupe_keys=_step2_dedupe_keys(intermediate_rows),
            canonical_base_columns=_canonical_columns_for_step(
                STEP2_EXPECTED_METRICS,
                include_round=has_round,
                include_replication=not has_round,
            ),
        )


def _row_has_any_keys(row: dict[str, object], keys: tuple[str, ...]) -> bool:
    return any(key in row for key in keys)


def _step2_dedupe_keys(rows: list[dict[str, object]]) -> tuple[str, ...]:
    """Construit les clés de déduplication Step2 selon les colonnes présentes."""
    dedupe_keys: list[str] = ["network_size", "algo", "snir_mode"]
    if any("round" in row for row in rows):
        dedupe_keys.append("round")
    elif any("replication" in row for row in rows):
        dedupe_keys.append("replication")
    if any("cluster" in row for row in rows):
        dedupe_keys.append("cluster")
    return tuple(dedupe_keys)


def write_step1_results(
    output_dir: Path,
    raw_rows: list[dict[str, object]],
    network_size: object | None = None,
    *,
    packet_rows: list[dict[str, object]] | None = None,
    metric_rows: list[dict[str, object]] | None = None,
) -> None:
    """Écrit raw_packets.csv, raw_metrics.csv et aggregated_results.csv pour l'étape 1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_replication_network_size(output_dir, raw_rows, step_label="step1")
    packets_path = output_dir / "raw_packets.csv"
    metrics_path = output_dir / "raw_metrics.csv"
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
        row["network_size"] = _coerce_positive_network_size(row_network_size)
        if row.get("density") not in (None, ""):
            row["density"] = _coerce_density(row["density"])

    packet_keys = ("node_id", "packet_id", "sf_selected")
    metric_keys = ("sent", "received", "pdr")
    if packet_rows is None:
        packet_rows = [
            row for row in raw_rows if _row_has_any_keys(row, packet_keys)
        ]
    if metric_rows is None:
        metric_rows = [
            row for row in raw_rows if _row_has_any_keys(row, metric_keys)
        ]

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
        _log_metric_summary(
            metric_rows,
            (
                "sent",
                "received",
                "pdr",
            ),
        )

    packets_header: list[str] = []
    seen: set[str] = set()
    if packet_rows:
        for row in packet_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    packets_header.append(key)
    else:
        packets_header = list(GROUP_KEYS)
    atomic_write_csv(
        packets_path,
        packets_header,
        [[row.get(key, "") for key in packets_header] for row in packet_rows],
    )

    metrics_header: list[str] = []
    seen = set()
    if metric_rows:
        for row in metric_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    metrics_header.append(key)
    else:
        metrics_header = list(GROUP_KEYS)
    atomic_write_csv(
        metrics_path,
        metrics_header,
        [[row.get(key, "") for key in metrics_header] for row in metric_rows],
    )

    _log_control_table(packet_rows, "raw_packets.csv")
    _log_control_table(metric_rows, "raw_metrics.csv")
    aggregated_rows, intermediate_rows = aggregate_results(
        metric_rows,
        expected_metrics=STEP1_EXPECTED_METRICS,
        step_label="Step1",
    )
    _log_control_table(aggregated_rows, "aggregated_results.csv")
    aggregated_header = (
        list(aggregated_rows[0].keys()) if aggregated_rows else list(BASE_GROUP_KEYS)
    )
    append_or_merge_csv(
        aggregated_path,
        aggregated_header,
        [[row.get(key, "") for key in aggregated_header] for row in aggregated_rows],
        dedupe_keys=MERGE_KEYS,
        canonical_base_columns=_canonical_columns_for_step(
            STEP1_EXPECTED_METRICS,
            include_round=False,
            include_replication=False,
        ),
    )
    if intermediate_rows:
        has_round = any(row.get("round") not in (None, "") for row in intermediate_rows)
        if has_round:
            intermediate_name = "aggregated_results_by_round.csv"
        else:
            intermediate_name = "aggregated_results_by_replication.csv"
        intermediate_path = output_dir / intermediate_name
        intermediate_header = list(intermediate_rows[0].keys())
        dedupe_keys = MERGE_KEYS + (("round",) if has_round else tuple())
        append_or_merge_csv(
            intermediate_path,
            intermediate_header,
            [
                [row.get(key, "") for key in intermediate_header]
                for row in intermediate_rows
            ],
            dedupe_keys=dedupe_keys,
            canonical_base_columns=_canonical_columns_for_step(
                STEP1_EXPECTED_METRICS,
                include_round=has_round,
                include_replication=not has_round,
            ),
        )
