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
        lowered = candidate.lower()
        if lowered in columns:
            return columns[lowered]
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "t"}:
        return True
    if lowered in {"0", "false", "no", "n", "f", ""}:
        return False
    return None


def _collect_network_sizes(
    rows: Iterable[dict[str, str]],
    *,
    candidates: Iterable[str] | None = None,
) -> tuple[list[float], str | None]:
    columns = _normalize_columns(rows)
    size_col = _pick_column(
        columns, candidates or ["network_size", "density", "num_nodes", "n_nodes"]
    )
    if not size_col:
        return [], None
    sizes = []
    for row in rows:
        value = _to_float(row.get(size_col))
        if value is None:
            continue
        sizes.append(value)
    return sizes, size_col


def _check_expected_network_sizes(
    rows: list[dict[str, str]],
    *,
    expected_sizes: set[int],
    label: str,
) -> CheckResult:
    sizes, size_col = _collect_network_sizes(rows)
    if not rows or not size_col:
        return CheckResult(label, "WARN", "Colonnes ou données manquantes.")
    found = {int(size) for size in sizes}
    missing = sorted(expected_sizes - found)
    extra = sorted(found - expected_sizes)
    if not missing and not extra:
        return CheckResult(label, "OK", f"Tailles trouvées: {sorted(found)}.")
    detail = (
        f"Tailles trouvées: {sorted(found)}; "
        f"manquantes={missing or 'aucune'}, "
        f"supplémentaires={extra or 'aucune'}."
    )
    return CheckResult(label, "WARN", detail)


def _check_step2_no_zero_network_size(rows: list[dict[str, str]]) -> CheckResult:
    sizes, size_col = _collect_network_sizes(rows, candidates=["network_size"])
    if not rows or not size_col:
        return CheckResult(
            "Step2 sans network_size à 0.0",
            "WARN",
            "Colonnes ou données manquantes.",
        )
    zero_count = sum(1 for size in sizes if float(size) == 0.0)
    status = "OK" if zero_count == 0 else "WARN"
    detail = f"Entrées à 0.0: {zero_count}."
    return CheckResult("Step2 sans network_size à 0.0", status, detail)


def _check_mixra_opt_fallback(rows: list[dict[str, str]]) -> CheckResult:
    columns = _normalize_columns(rows)
    algo_col = _pick_column(columns, ["algo", "algorithm"])
    fallback_col = _pick_column(
        columns, ["mixra_opt_fallback", "mixra_fallback", "fallback"]
    )

    if not rows or not algo_col or not fallback_col:
        return CheckResult(
            "MixRA-Opt fallback False",
            "WARN",
            "Colonnes ou données manquantes.",
        )

    total = 0
    true_count = 0
    for row in rows:
        algo = row.get(algo_col)
        if algo is None:
            continue
        algo_key = str(algo).strip().lower()
        if algo_key not in {"mixra_opt", "mixra-opt", "mixra opt"}:
            continue
        fallback = _to_bool(row.get(fallback_col))
        if fallback is None:
            continue
        total += 1
        if fallback:
            true_count += 1

    if total == 0:
        return CheckResult(
            "MixRA-Opt fallback False",
            "WARN",
            "Aucune ligne MixRA-Opt exploitable.",
        )

    status = "OK" if true_count == 0 else "WARN"
    detail = f"Lignes MixRA-Opt: {total}, fallback True: {true_count}."
    return CheckResult("MixRA-Opt fallback False", status, detail)


def _report(results: list[CheckResult]) -> None:
    print("Validation des résultats agrégés")
    print("=" * 35)
    for result in results:
        print(f"- {result.name}: {result.status} ({result.details})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valide les points clés dans aggregated_results.csv.",
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
    args = parser.parse_args()

    step1_rows = _read_rows(args.step1)
    step2_rows = _read_rows(args.step2)
    expected_sizes = {80, 160, 320, 640, 1280}

    results = [
        _check_expected_network_sizes(
            step1_rows,
            expected_sizes=expected_sizes,
            label="Tailles réseau attendues (step1)",
        ),
        _check_expected_network_sizes(
            step2_rows,
            expected_sizes=expected_sizes,
            label="Tailles réseau attendues (step2)",
        ),
        _check_mixra_opt_fallback(step1_rows),
        _check_step2_no_zero_network_size(step2_rows),
    ]

    _report(results)


if __name__ == "__main__":
    main()
