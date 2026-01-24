"""Génère un rapport d'intégrité pour les résultats de l'article C."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from article_c.common.csv_io import _normalize_group_keys, _normalize_snir_mode


PDR_KEY_GROUPS = (
    ("sent", "received", "pdr"),
    ("sent_mean", "received_mean", "pdr_mean"),
)


@dataclass
class GroupStats:
    algo: str
    snir_mode: str
    cluster: str
    row_count: int = 0
    sizes: set[str] = field(default_factory=set)
    pdr_checked: int = 0
    pdr_issues: int = 0


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_size(value: object) -> str | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}"


def _normalize_algo(value: object) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def _normalize_cluster(value: object) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def _infer_snir_mode(row: dict[str, object]) -> str:
    snir_mode = _normalize_snir_mode(row.get("snir_mode"))
    if snir_mode is None:
        snir_mode = _normalize_snir_mode(row.get("snir_state"))
    if snir_mode is None:
        snir_mode = _normalize_snir_mode(row.get("snir"))
    return snir_mode or "snir_unknown"


def _check_pdr_row(
    row: dict[str, object],
    sent_key: str,
    received_key: str,
    pdr_key: str,
    tolerance: float,
) -> bool | None:
    if not {sent_key, received_key, pdr_key}.issubset(row.keys()):
        return None
    sent = _parse_float(row.get(sent_key))
    received = _parse_float(row.get(received_key))
    pdr = _parse_float(row.get(pdr_key))
    if sent is None or received is None or pdr is None:
        return None
    expected = sent * pdr
    diff = abs(received - expected)
    limit = max(1.0, abs(expected)) * tolerance
    return diff <= limit


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collect_stats(
    rows: list[dict[str, object]],
    tolerance: float,
) -> tuple[dict[tuple[str, str, str], GroupStats], set[str]]:
    if not rows:
        return {}, set()
    _normalize_group_keys(rows)
    groups: dict[tuple[str, str, str], GroupStats] = {}
    all_sizes: set[str] = set()
    for row in rows:
        algo = _normalize_algo(row.get("algo") or row.get("algorithm"))
        snir_mode = _infer_snir_mode(row)
        cluster = _normalize_cluster(row.get("cluster"))
        key = (algo, snir_mode, cluster)
        if key not in groups:
            groups[key] = GroupStats(algo=algo, snir_mode=snir_mode, cluster=cluster)
        stats = groups[key]
        stats.row_count += 1
        size_label = _normalize_size(row.get("network_size") or row.get("density"))
        if size_label:
            stats.sizes.add(size_label)
            all_sizes.add(size_label)
        for sent_key, received_key, pdr_key in PDR_KEY_GROUPS:
            verdict = _check_pdr_row(row, sent_key, received_key, pdr_key, tolerance)
            if verdict is None:
                continue
            stats.pdr_checked += 1
            if not verdict:
                stats.pdr_issues += 1
            break
    return groups, all_sizes


def _format_list(values: set[str]) -> str:
    if not values:
        return "-"
    return ";".join(sorted(values, key=lambda item: (len(item), item)))


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    fmt = " | ".join(f"{{:<{width}}}" for width in widths)
    separator = "-+-".join("-" * width for width in widths)
    print(fmt.format(*headers))
    print(separator)
    for row in rows:
        print(fmt.format(*row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Génère un rapport d'intégrité (comptage, cohérence PDR, tailles manquantes)."
        )
    )
    parser.add_argument(
        "--step1-dir",
        type=Path,
        default=Path("article_c/step1/results"),
        help="Répertoire des résultats de l'étape 1.",
    )
    parser.add_argument(
        "--step2-dir",
        type=Path,
        default=Path("article_c/step2/results"),
        help="Répertoire des résultats de l'étape 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("article_c/report_integrity.csv"),
        help="Chemin du CSV de sortie.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolérance relative pour received ≈ sent*pdr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = [
        args.step1_dir / "raw_metrics.csv",
        args.step1_dir / "aggregated_results.csv",
        args.step2_dir / "raw_results.csv",
        args.step2_dir / "aggregated_results.csv",
    ]
    report_rows: list[list[str]] = []
    console_rows: list[list[str]] = []
    for path in files:
        rows = _read_csv(path)
        groups, all_sizes = _collect_stats(rows, args.tolerance)
        if not rows:
            report_rows.append(
                [
                    str(path),
                    "n/a",
                    "n/a",
                    "n/a",
                    "0",
                    "-",
                    "-",
                    "0",
                    "0",
                ]
            )
            console_rows.append(
                [
                    str(path),
                    "n/a",
                    "n/a",
                    "n/a",
                    "0",
                    "-",
                    "-",
                    "0",
                    "0",
                ]
            )
            continue
        for stats in sorted(
            groups.values(),
            key=lambda item: (item.algo, item.snir_mode, item.cluster),
        ):
            missing_sizes = all_sizes - stats.sizes
            report_rows.append(
                [
                    str(path),
                    stats.algo,
                    stats.snir_mode,
                    stats.cluster,
                    str(stats.row_count),
                    _format_list(stats.sizes),
                    _format_list(missing_sizes),
                    str(stats.pdr_issues),
                    str(stats.pdr_checked),
                ]
            )
            console_rows.append(
                [
                    Path(path).name,
                    stats.algo,
                    stats.snir_mode,
                    stats.cluster,
                    str(stats.row_count),
                    _format_list(missing_sizes),
                    f"{stats.pdr_issues}/{stats.pdr_checked}",
                ]
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "source_file",
        "algo",
        "snir_mode",
        "cluster",
        "row_count",
        "sizes_present",
        "missing_sizes",
        "pdr_issues",
        "pdr_checked",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(report_rows)

    print("Rapport d'intégrité généré.")
    _print_table(
        console_rows,
        [
            "source",
            "algo",
            "snir_mode",
            "cluster",
            "rows",
            "tailles manquantes",
            "pdr (err/ok)",
        ],
    )
    print(f"CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
