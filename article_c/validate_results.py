"""Validation rapide des résultats agrégés de l'article C."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import pvariance
from typing import Iterable


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _normalize_columns(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for row in rows:
        for key in row.keys():
            lowered = key.strip().lower()
            if lowered not in columns:
                columns[lowered] = key
    return columns


def _pick_column(columns: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snir_state(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"snir_on", "on", "true", "1", "yes"}:
        return "snir_on"
    if lowered in {"snir_off", "off", "false", "0", "no"}:
        return "snir_off"
    return lowered


def _slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def _check_snir_pdr(rows: list[dict[str, str]]) -> CheckResult:
    columns = _normalize_columns(rows)
    pdr_col = _pick_column(columns, ["pdr_mean", "pdr", "PDR_mean", "PDR"])
    snir_col = _pick_column(columns, ["snir_mode", "snir_state", "with_snir"])
    density_col = _pick_column(columns, ["density", "num_nodes", "n_nodes"])
    algo_col = _pick_column(columns, ["algo", "algorithm"])
    cluster_col = _pick_column(columns, ["cluster"])

    if not rows or not pdr_col or not snir_col or not density_col:
        return CheckResult(
            "PDR SNIR ON < SNIR OFF",
            "N/A",
            "Colonnes ou données manquantes.",
        )

    comparisons = 0
    failures = 0
    grouped: dict[tuple[str, float], dict[str, float]] = {}

    for row in rows:
        if cluster_col and row.get(cluster_col) not in {"", None, "all"}:
            continue
        density = _to_float(row.get(density_col))
        pdr = _to_float(row.get(pdr_col))
        if density is None or pdr is None:
            continue
        algo = row.get(algo_col, "") if algo_col else ""
        snir_state = _snir_state(row.get(snir_col))
        if snir_state not in {"snir_on", "snir_off"}:
            continue
        key = (algo, density)
        grouped.setdefault(key, {})[snir_state] = pdr

    for values in grouped.values():
        if "snir_on" in values and "snir_off" in values:
            comparisons += 1
            if values["snir_on"] >= values["snir_off"]:
                failures += 1

    if comparisons == 0:
        return CheckResult(
            "PDR SNIR ON < SNIR OFF",
            "N/A",
            "Aucune paire comparable trouvée.",
        )

    status = "OK" if failures == 0 else "KO"
    detail = f"Comparaisons: {comparisons}, échecs: {failures}."
    return CheckResult("PDR SNIR ON < SNIR OFF", status, detail)


def _check_trend(
    rows: list[dict[str, str]],
    *,
    metric_candidates: Iterable[str],
    trend: str,
    label: str,
) -> CheckResult:
    columns = _normalize_columns(rows)
    metric_col = _pick_column(columns, metric_candidates)
    node_col = _pick_column(columns, ["num_nodes", "n_nodes", "density"])
    algo_col = _pick_column(columns, ["algo", "algorithm"])
    snir_col = _pick_column(columns, ["snir_mode", "snir_state", "with_snir"])
    cluster_col = _pick_column(columns, ["cluster"])

    if not rows or not metric_col or not node_col:
        return CheckResult(label, "N/A", "Colonnes ou données manquantes.")

    grouped: dict[tuple[str, str], dict[float, list[float]]] = {}
    for row in rows:
        if cluster_col and row.get(cluster_col) not in {"", None, "all"}:
            continue
        node_count = _to_float(row.get(node_col))
        metric_value = _to_float(row.get(metric_col))
        if node_count is None or metric_value is None:
            continue
        algo = row.get(algo_col, "") if algo_col else ""
        snir_state = _snir_state(row.get(snir_col)) or ""
        key = (algo, snir_state)
        grouped.setdefault(key, {}).setdefault(node_count, []).append(metric_value)

    checked = 0
    failures = 0
    for values in grouped.values():
        points = sorted((node, sum(vals) / len(vals)) for node, vals in values.items())
        if len(points) < 2:
            continue
        xs = [node for node, _ in points]
        ys = [val for _, val in points]
        slope = _slope(xs, ys)
        if slope is None:
            continue
        checked += 1
        if trend == "down" and slope >= 0:
            failures += 1
        if trend == "up" and slope <= 0:
            failures += 1

    if checked == 0:
        return CheckResult(label, "N/A", "Pas assez de points pour évaluer la tendance.")

    status = "OK" if failures == 0 else "KO"
    detail = f"Groupes analysés: {checked}, échecs: {failures}."
    return CheckResult(label, status, detail)


def _check_rl1_variance(rows: list[dict[str, str]], threshold: float) -> CheckResult:
    columns = _normalize_columns(rows)
    reward_col = _pick_column(columns, ["reward_mean", "reward"])
    density_col = _pick_column(columns, ["density", "num_nodes", "n_nodes"])
    algo_col = _pick_column(columns, ["algo", "algorithm"])
    snir_col = _pick_column(columns, ["snir_mode", "snir_state", "with_snir"])
    cluster_col = _pick_column(columns, ["cluster"])

    if not rows or not reward_col or not density_col:
        return CheckResult("RL1 non plate", "N/A", "Colonnes ou données manquantes.")

    grouped: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        if cluster_col and row.get(cluster_col) not in {"", None, "all"}:
            continue
        if snir_col:
            snir_state = _snir_state(row.get(snir_col))
            if snir_state not in {"snir_on", None, ""}:
                continue
        density = _to_float(row.get(density_col))
        reward = _to_float(row.get(reward_col))
        if density is None or reward is None:
            continue
        algo = row.get(algo_col, "") if algo_col else ""
        grouped.setdefault(algo, {}).setdefault(density, []).append(reward)

    checked = 0
    failures = 0
    for values in grouped.values():
        points = [sum(vals) / len(vals) for _, vals in sorted(values.items())]
        if len(points) < 2:
            continue
        checked += 1
        if pvariance(points) <= threshold:
            failures += 1

    if checked == 0:
        return CheckResult("RL1 non plate", "N/A", "Pas assez de points pour la variance.")

    status = "OK" if failures == 0 else "KO"
    detail = (
        f"Algorithmes analysés: {checked}, échecs: {failures}, seuil={threshold}"
    )
    return CheckResult("RL1 non plate", status, detail)


def _report(results: list[CheckResult]) -> None:
    print("Validation des résultats agrégés")
    print("=" * 35)
    for result in results:
        print(f"- {result.name}: {result.status} ({result.details})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valide les tendances clés dans aggregated_results.csv.",
    )
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--step1",
        type=Path,
        default=base_dir / "step1" / "results" / "aggregated_results.csv",
        help="Chemin vers aggregated_results.csv de l'étape 1.",
    )
    parser.add_argument(
        "--step2",
        type=Path,
        default=base_dir / "step2" / "results" / "aggregated_results.csv",
        help="Chemin vers aggregated_results.csv de l'étape 2.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-4,
        help="Seuil minimal de variance pour la courbe RL1.",
    )
    args = parser.parse_args()

    step1_rows = _read_rows(args.step1)
    step2_rows = _read_rows(args.step2)

    results = [
        _check_snir_pdr(step1_rows),
        _check_trend(
            step1_rows,
            metric_candidates=[
                "goodput_mean",
                "goodput_bps_mean",
                "throughput_bps_mean",
                "throughput_mean",
                "goodput",
                "throughput_bps",
            ],
            trend="down",
            label="Goodput diminue avec les nœuds",
        ),
        _check_trend(
            step1_rows,
            metric_candidates=[
                "collision_rate_mean",
                "collision_rate",
                "collisions_mean",
                "collisions",
                "collisions_snir_mean",
            ],
            trend="up",
            label="Collision rate augmente",
        ),
        _check_rl1_variance(step2_rows, args.variance_threshold),
    ]

    _report(results)


if __name__ == "__main__":
    main()
