"""Validation rapide des résultats agrégés de l'article C."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
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
            "WARN",
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
            "WARN",
            "Aucune paire comparable trouvée.",
        )

    status = "OK" if failures == 0 else "WARN"
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
        return CheckResult(label, "WARN", "Colonnes ou données manquantes.")

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
        return CheckResult(label, "WARN", "Pas assez de points pour évaluer la tendance.")

    status = "OK" if failures == 0 else "WARN"
    detail = f"Groupes analysés: {checked}, échecs: {failures}."
    return CheckResult(label, status, detail)


def _check_reward_rise_plateau(
    rows: list[dict[str, str]],
    *,
    slope_threshold: float,
    plateau_threshold: float,
) -> CheckResult:
    columns = _normalize_columns(rows)
    reward_col = _pick_column(columns, ["reward_mean", "reward"])
    density_col = _pick_column(columns, ["density", "num_nodes", "n_nodes"])
    algo_col = _pick_column(columns, ["algo", "algorithm"])
    snir_col = _pick_column(columns, ["snir_mode", "snir_state", "with_snir"])
    cluster_col = _pick_column(columns, ["cluster"])

    if not rows or not reward_col or not density_col:
        return CheckResult(
            "Reward RL monte puis se stabilise",
            "WARN",
            "Colonnes ou données manquantes.",
        )

    grouped: dict[tuple[str, str], dict[float, list[float]]] = {}
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
        snir_state = _snir_state(row.get(snir_col)) or "" if snir_col else ""
        grouped.setdefault((algo, snir_state), {}).setdefault(density, []).append(
            reward
        )

    checked = 0
    failures = 0
    for values in grouped.values():
        points = [sum(vals) / len(vals) for _, vals in sorted(values.items())]
        if len(points) < 4:
            continue
        mid = max(2, len(points) // 2)
        early_xs = list(range(mid))
        early_ys = points[:mid]
        late_xs = list(range(len(points) - mid))
        late_ys = points[-(len(points) - mid) :]
        early_slope = _slope(early_xs, early_ys)
        late_slope = _slope(late_xs, late_ys)
        if early_slope is None or late_slope is None:
            continue
        checked += 1
        if early_slope <= slope_threshold or abs(late_slope) > plateau_threshold:
            failures += 1

    if checked == 0:
        return CheckResult(
            "Reward RL monte puis se stabilise",
            "WARN",
            "Pas assez de points pour analyser la courbe.",
        )

    status = "OK" if failures == 0 else "WARN"
    detail = (
        "Groupes analysés: "
        f"{checked}, échecs: {failures}, pente_min={slope_threshold}, plateau={plateau_threshold}"
    )
    return CheckResult("Reward RL monte puis se stabilise", status, detail)


def _check_cluster_order(
    rows: list[dict[str, str]],
    *,
    metric_candidates: Iterable[str],
    label: str,
) -> CheckResult:
    columns = _normalize_columns(rows)
    cluster_col = _pick_column(columns, ["cluster"])
    metric_col = _pick_column(columns, metric_candidates)

    if not rows or not cluster_col or not metric_col:
        return CheckResult(label, "WARN", "Colonnes ou données manquantes.")

    cluster_values: dict[str, list[float]] = {"C1": [], "C2": [], "C3": []}
    for row in rows:
        cluster = row.get(cluster_col)
        if cluster is None:
            continue
        cluster_key = str(cluster).strip().upper()
        if cluster_key not in cluster_values:
            continue
        metric_value = _to_float(row.get(metric_col))
        if metric_value is None:
            continue
        cluster_values[cluster_key].append(metric_value)

    if any(not values for values in cluster_values.values()):
        return CheckResult(label, "WARN", "Clusters manquants ou sans données.")

    means = {key: sum(vals) / len(vals) for key, vals in cluster_values.items()}
    ok = means["C1"] > means["C2"] > means["C3"]
    status = "OK" if ok else "WARN"
    detail = (
        "Moyennes: "
        f"C1={means['C1']:.4g}, C2={means['C2']:.4g}, C3={means['C3']:.4g}."
    )
    return CheckResult(label, status, detail)


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
        "--slope-threshold",
        type=float,
        default=0.0,
        help="Pente minimale attendue pour la phase montante du reward RL.",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-3,
        help="Seuil maximal pour considérer la phase plateau du reward RL.",
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
            label="Collision rate augmente avec N",
        ),
        _check_reward_rise_plateau(
            step2_rows,
            slope_threshold=args.slope_threshold,
            plateau_threshold=args.plateau_threshold,
        ),
        _check_cluster_order(
            step1_rows or step2_rows,
            metric_candidates=[
                "pdr_mean",
                "pdr",
                "goodput_mean",
                "throughput_mean",
                "reward_mean",
                "reward",
            ],
            label="Cluster C1 > C2 > C3",
        ),
    ]

    _report(results)


if __name__ == "__main__":
    main()
