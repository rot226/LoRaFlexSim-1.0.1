"""Contrôles QA scientifiques avant génération des figures de l'article C."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


DEFAULT_STEP1_CSV = Path("article_c/step1/results/aggregated_results.csv")
DEFAULT_STEP2_CSV = Path("article_c/step2/results/aggregated_results.csv")
DEFAULT_REPORT_TXT = Path("article_c/scientific_qa_report.txt")
DEFAULT_REPORT_CSV = Path("article_c/scientific_qa_report.csv")

SUCCESS_CANDIDATES = (
    "success_rate_mean",
    "pdr_mean",
    "success_rate",
    "pdr",
)
ENERGY_CANDIDATES = (
    "energy_per_delivered_packet_mean",
    "energy_per_packet_mean",
    "energy_per_success_mean",
    "energy_per_delivered_packet_j",
    "energy_per_success",
)


@dataclass
class CheckResult:
    check_id: str
    scope: str
    verdict: str
    details: str
    warnings: str = ""


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _read_csv(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader]
    return rows, fieldnames


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _row_group_key(row: dict[str, object]) -> tuple[str, str, str]:
    algo = str(row.get("algo") or row.get("algorithm") or "all").strip().lower() or "all"
    snir_mode = str(row.get("snir_mode") or row.get("snir") or "all").strip().lower() or "all"
    cluster = str(row.get("cluster") or "all").strip().lower() or "all"
    return algo, snir_mode, cluster


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _check_no_nan_inf(rows: list[dict[str, object]], fieldnames: list[str], scope: str) -> CheckResult:
    if not fieldnames:
        return CheckResult(
            check_id="nan_inf_absence",
            scope=scope,
            verdict="WARN",
            details="CSV absent ou vide, contrôle non exécutable.",
        )
    non_finite_hits: list[str] = []
    for row_idx, row in enumerate(rows, start=2):
        for col in fieldnames:
            parsed = _parse_float(row.get(col))
            if parsed is None:
                continue
            if not math.isfinite(parsed):
                non_finite_hits.append(f"ligne {row_idx}, colonne {col}")
                if len(non_finite_hits) >= 10:
                    break
        if len(non_finite_hits) >= 10:
            break
    if non_finite_hits:
        return CheckResult(
            check_id="nan_inf_absence",
            scope=scope,
            verdict="FAIL",
            details="Valeurs NaN/Inf détectées.",
            warnings="; ".join(non_finite_hits),
        )
    return CheckResult(
        check_id="nan_inf_absence",
        scope=scope,
        verdict="PASS",
        details="Aucune valeur NaN/Inf détectée.",
    )


def _check_success_vs_density(rows: list[dict[str, object]], fieldnames: list[str], scope: str) -> CheckResult:
    success_col = _pick_column(fieldnames, SUCCESS_CANDIDATES)
    if success_col is None:
        return CheckResult(
            check_id="success_vs_density",
            scope=scope,
            verdict="WARN",
            details="Aucune métrique de succès détectée.",
        )
    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    for row in rows:
        size_val = _parse_float(row.get("network_size") or row.get("density"))
        success_val = _parse_float(row.get(success_col))
        if size_val is None or success_val is None:
            continue
        if not math.isfinite(size_val) or not math.isfinite(success_val):
            continue
        group = grouped.setdefault(_row_group_key(row), {})
        group.setdefault(int(round(size_val)), []).append(success_val)

    severe_violations: list[str] = []
    local_bumps = 0
    checked_groups = 0
    for group_key, by_size in grouped.items():
        if len(by_size) < 2:
            continue
        checked_groups += 1
        points = sorted((size, _mean(values)) for size, values in by_size.items())
        first_size, first_value = points[0]
        last_size, last_value = points[-1]
        if last_value > first_value + 0.02:
            severe_violations.append(
                f"{group_key}: succès {first_size}->{last_size} ({first_value:.4f}->{last_value:.4f})"
            )
        for (_, prev), (_, curr) in zip(points, points[1:]):
            if curr > prev + 0.01:
                local_bumps += 1

    if checked_groups == 0:
        return CheckResult(
            check_id="success_vs_density",
            scope=scope,
            verdict="WARN",
            details="Pas assez de points (>=2 densités) pour vérifier la tendance.",
        )
    if severe_violations:
        return CheckResult(
            check_id="success_vs_density",
            scope=scope,
            verdict="FAIL",
            details="Tendance décroissante violée pour au moins un groupe.",
            warnings=" | ".join(severe_violations[:5]),
        )
    if local_bumps > 0:
        return CheckResult(
            check_id="success_vs_density",
            scope=scope,
            verdict="WARN",
            details=(
                "Tendance globale décroissante respectée, mais des hausses locales "
                "ont été détectées."
            ),
            warnings=f"hausses locales détectées: {local_bumps}",
        )
    return CheckResult(
        check_id="success_vs_density",
        scope=scope,
        verdict="PASS",
        details="Succès décroissant avec la densité (sur les groupes exploitables).",
    )


def _check_energy_vs_success(rows: list[dict[str, object]], fieldnames: list[str], scope: str) -> CheckResult:
    success_col = _pick_column(fieldnames, SUCCESS_CANDIDATES)
    energy_col = _pick_column(fieldnames, ENERGY_CANDIDATES)
    if success_col is None or energy_col is None:
        return CheckResult(
            check_id="energy_vs_success",
            scope=scope,
            verdict="WARN",
            details=(
                "Colonnes énergie/succès insuffisantes pour le contrôle "
                f"(success={success_col}, energy={energy_col})."
            ),
        )

    weighted_delta = 0.0
    incoherences = 0
    segments = 0
    for by_size in _iter_grouped_curves(rows, success_col, energy_col).values():
        if len(by_size) < 2:
            continue
        points = sorted(by_size.items())
        for (_, prev_values), (_, curr_values) in zip(points, points[1:]):
            prev_success, prev_energy = prev_values
            curr_success, curr_energy = curr_values
            d_success = curr_success - prev_success
            d_energy = curr_energy - prev_energy
            segments += 1
            if d_success >= 0.02 and d_energy > 0.0:
                incoherences += 1
            weighted_delta += d_success * d_energy

    if segments == 0:
        return CheckResult(
            check_id="energy_vs_success",
            scope=scope,
            verdict="WARN",
            details="Pas assez de points pour vérifier énergie vs succès.",
        )
    if incoherences > 0:
        return CheckResult(
            check_id="energy_vs_success",
            scope=scope,
            verdict="WARN",
            details="Des incohérences locales énergie/succès ont été détectées.",
            warnings=f"segments incohérents: {incoherences}/{segments}",
        )
    if weighted_delta > 0:
        return CheckResult(
            check_id="energy_vs_success",
            scope=scope,
            verdict="WARN",
            details=(
                "Relation énergie/succès peu nette (covariance locale positive)."
            ),
            warnings=f"score={weighted_delta:.6f}",
        )
    return CheckResult(
        check_id="energy_vs_success",
        scope=scope,
        verdict="PASS",
        details="Cohérence énergie vs succès/PDR globalement respectée.",
    )


def _iter_grouped_curves(
    rows: list[dict[str, object]],
    success_col: str,
    energy_col: str,
) -> dict[tuple[str, str, str], dict[int, tuple[float, float]]]:
    grouped: dict[tuple[str, str, str], dict[int, list[tuple[float, float]]]] = {}
    for row in rows:
        size_val = _parse_float(row.get("network_size") or row.get("density"))
        success_val = _parse_float(row.get(success_col))
        energy_val = _parse_float(row.get(energy_col))
        if size_val is None or success_val is None or energy_val is None:
            continue
        if not (math.isfinite(size_val) and math.isfinite(success_val) and math.isfinite(energy_val)):
            continue
        grouped.setdefault(_row_group_key(row), {}).setdefault(int(round(size_val)), []).append(
            (success_val, energy_val)
        )

    curves: dict[tuple[str, str, str], dict[int, tuple[float, float]]] = {}
    for group_key, by_size in grouped.items():
        curves[group_key] = {
            size: (_mean([v[0] for v in values]), _mean([v[1] for v in values]))
            for size, values in by_size.items()
        }
    return curves


def run_scientific_checks(
    *,
    step1_csv: Path,
    step2_csv: Path,
    report_txt: Path,
    report_csv: Path,
) -> tuple[int, list[CheckResult]]:
    reports: list[CheckResult] = []
    for scope, path in (("step1", step1_csv), ("step2", step2_csv)):
        rows, fieldnames = _read_csv(path)
        reports.append(_check_no_nan_inf(rows, fieldnames, scope))
        reports.append(_check_success_vs_density(rows, fieldnames, scope))
        reports.append(_check_energy_vs_success(rows, fieldnames, scope))

    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "scope", "verdict", "details", "warnings"])
        for item in reports:
            writer.writerow([item.check_id, item.scope, item.verdict, item.details, item.warnings])

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for item in reports:
        counts[item.verdict] = counts.get(item.verdict, 0) + 1

    lines = [
        "Rapport QA scientifique",
        "======================",
        f"Step1 CSV: {step1_csv}",
        f"Step2 CSV: {step2_csv}",
        "",
    ]
    for item in reports:
        lines.append(f"[{item.verdict}] {item.scope}::{item.check_id} - {item.details}")
        if item.warnings:
            lines.append(f"  warning: {item.warnings}")
    lines.append("")
    lines.append(
        "Bilan: "
        f"{counts['PASS']} PASS / {counts['WARN']} WARN / {counts['FAIL']} FAIL"
    )
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Rapport QA scientifique écrit: {report_txt}")
    print(f"Rapport QA scientifique CSV: {report_csv}")
    print(lines[-1])

    exit_code = 1 if counts["FAIL"] > 0 else 0
    return exit_code, reports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vérifie la cohérence scientifique des résultats (article C)."
    )
    parser.add_argument("--step1-csv", type=Path, default=DEFAULT_STEP1_CSV)
    parser.add_argument("--step2-csv", type=Path, default=DEFAULT_STEP2_CSV)
    parser.add_argument("--report-txt", type=Path, default=DEFAULT_REPORT_TXT)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exit_code, _ = run_scientific_checks(
        step1_csv=args.step1_csv,
        step2_csv=args.step2_csv,
        report_txt=args.report_txt,
        report_csv=args.report_csv,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
