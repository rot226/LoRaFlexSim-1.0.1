#!/usr/bin/env python3
"""Sanity checks pour les sorties QoS (SNIR, PDR, distributions)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class CheckIssue:
    level: str
    message: str


def _parse_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _load_json(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _weighted_variance(values: dict[float, float]) -> float | None:
    weights = [float(w) for w in values.values() if w is not None]
    if not weights:
        return None
    total = sum(weights)
    if total <= 0:
        return None
    mean = sum(float(v) * float(w) for v, w in values.items()) / total
    variance = sum(float(w) * (float(v) - mean) ** 2 for v, w in values.items()) / total
    return variance


def _coerce_distribution(raw: dict[str, Any] | None) -> dict[float, float] | None:
    if not raw:
        return None
    output: dict[float, float] = {}
    for key, value in raw.items():
        try:
            output[float(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output or None


def _iter_csv_files(targets: Iterable[Path]) -> list[Path]:
    csv_files: list[Path] = []
    for target in targets:
        if target.is_file() and target.suffix.lower() == ".csv":
            csv_files.append(target)
        elif target.is_dir():
            csv_files.extend(sorted(target.rglob("*.csv")))
    return csv_files


def _snir_state(row: dict[str, str]) -> str | None:
    state = (row.get("snir_state") or "").strip().lower()
    if state in {"snir_on", "snir_off"}:
        return state
    flag = _parse_bool(row.get("with_snir") or row.get("use_snir"))
    if flag is True:
        return "snir_on"
    if flag is False:
        return "snir_off"
    return None


def _float_or_zero(value: str | None) -> float:
    parsed = _parse_float(value, default=0.0)
    return 0.0 if parsed is None else float(parsed)


def _int_or_zero(value: str | None) -> int:
    parsed = _parse_int(value, default=0)
    return 0 if parsed is None else int(parsed)


def run_checks(
    csv_files: list[Path],
    *,
    epsilon: float,
    large_nodes: int,
    high_pdr_threshold: float,
) -> tuple[list[CheckIssue], list[CheckIssue]]:
    warnings: list[CheckIssue] = []
    failures: list[CheckIssue] = []
    if not csv_files:
        failures.append(CheckIssue("FAIL", "Aucun CSV trouvé pour l'analyse."))
        return warnings, failures

    records: list[dict[str, Any]] = []
    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row["__csv_path"] = str(csv_path)
                    records.append(row)
        except OSError as exc:
            failures.append(CheckIssue("FAIL", f"Lecture impossible de {csv_path}: {exc}"))

    if not records:
        failures.append(CheckIssue("FAIL", "Aucune ligne détectée dans les CSV."))
        return warnings, failures

    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"snir_on": [], "snir_off": []}
    )
    for row in records:
        key = (
            (row.get("algorithm") or "").strip() or None,
            _int_or_zero(row.get("num_nodes")),
            _parse_float(row.get("packet_interval_s"), default=0.0) or 0.0,
            _parse_float(row.get("simulation_duration_s"), default=0.0) or 0.0,
        )
        state = _snir_state(row)
        if state in {"snir_on", "snir_off"}:
            grouped[key][state].append(row)

    for key, bundle in sorted(grouped.items()):
        if not bundle["snir_on"] or not bundle["snir_off"]:
            continue
        metric_pairs = {
            "PDR": (bundle["snir_on"], bundle["snir_off"]),
            "throughput_bps": (bundle["snir_on"], bundle["snir_off"]),
            "collisions": (bundle["snir_on"], bundle["snir_off"]),
        }
        for metric, (rows_on, rows_off) in metric_pairs.items():
            values_on = [_float_or_zero(r.get(metric)) for r in rows_on]
            values_off = [_float_or_zero(r.get(metric)) for r in rows_off]
            if not values_on or not values_off:
                warnings.append(
                    CheckIssue(
                        "WARN",
                        f"Valeurs manquantes pour {metric} dans le groupe {key}.",
                    )
                )
                continue
            mean_on = sum(values_on) / len(values_on)
            mean_off = sum(values_off) / len(values_off)
            delta = abs(mean_on - mean_off)
            if delta <= epsilon:
                failures.append(
                    CheckIssue(
                        "FAIL",
                        f"Δ {metric} trop faible ({delta:.4f} ≤ {epsilon:.4f}) pour {key}.",
                    )
                )

    for row in records:
        num_nodes = _int_or_zero(row.get("num_nodes"))
        if num_nodes < large_nodes:
            continue
        pdr = _parse_float(row.get("PDR"))
        der = _parse_float(row.get("DER"))
        if pdr is not None and pdr > high_pdr_threshold:
            warnings.append(
                CheckIssue(
                    "WARN",
                    f"PDR élevée (>{high_pdr_threshold}) pour N={num_nodes} ({row.get('__csv_path')}).",
                )
            )
        if der is not None and der > high_pdr_threshold:
            warnings.append(
                CheckIssue(
                    "WARN",
                    f"DER élevée (>{high_pdr_threshold}) pour N={num_nodes} ({row.get('__csv_path')}).",
                )
            )

        sf_dist = _coerce_distribution(_load_json(row.get("sf_distribution_json")))
        snr_dist = _coerce_distribution(_load_json(row.get("snr_histogram_json")))
        snir_dist = _coerce_distribution(_load_json(row.get("snir_histogram_json")))
        collision_breakdown = _load_json(row.get("collision_breakdown_json"))
        collision_dist: dict[float, float] | None = None
        if collision_breakdown:
            collision_dist = _coerce_distribution(collision_breakdown.get("by_sf"))
            if not collision_dist:
                collision_dist = _coerce_distribution(collision_breakdown.get("by_channel"))

        dist_map = {
            "SF": sf_dist,
            "SNR": snr_dist,
            "SNIR": snir_dist,
            "collisions": collision_dist,
        }
        for label, dist in dist_map.items():
            if dist is None:
                warnings.append(
                    CheckIssue(
                        "WARN",
                        f"Distribution {label} manquante ({row.get('__csv_path')}).",
                    )
                )
                continue
            variance = _weighted_variance(dist)
            if variance is None:
                warnings.append(
                    CheckIssue(
                        "WARN",
                        f"Distribution {label} vide ({row.get('__csv_path')}).",
                    )
                )
                continue
            if variance <= 0.0:
                failures.append(
                    CheckIssue(
                        "FAIL",
                        f"Variance nulle pour {label} ({row.get('__csv_path')}).",
                    )
                )

    jain_values = [
        _parse_float(row.get("jain_index"))
        for row in records
        if row.get("jain_index") not in (None, "")
    ]
    if jain_values and all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9) for value in jain_values):
        warnings.append(CheckIssue("WARN", "Indice de Jain == 1.0 pour toutes les lignes."))

    return warnings, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanity checks SNIR/PDR sur les CSV QoS.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("results")],
        help="Fichiers/dossiers CSV à analyser (défaut: results).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Écart minimal attendu entre SNIR_ON et SNIR_OFF.",
    )
    parser.add_argument(
        "--large-nodes",
        type=int,
        default=100,
        help="Seuil N considéré comme 'grand' pour PDR/DER.",
    )
    parser.add_argument(
        "--pdr-der-threshold",
        type=float,
        default=0.999,
        help="Seuil PDR/DER déclenchant une alerte.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Retourne un code d'erreur si des WARN sont présents.",
    )
    args = parser.parse_args(argv)

    csv_files = _iter_csv_files(args.paths)
    warnings, failures = run_checks(
        csv_files,
        epsilon=args.epsilon,
        large_nodes=args.large_nodes,
        high_pdr_threshold=args.pdr_der_threshold,
    )

    for issue in warnings + failures:
        print(f"[{issue.level}] {issue.message}")

    if failures or (warnings and args.fail_on_warn):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
