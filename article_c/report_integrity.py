"""Génère un rapport d'intégrité pour les résultats de l'article C."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

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


@dataclass
class IntegrityIssueCounter:
    missing_files: int = 0
    replication_holes: int = 0
    column_inconsistencies: int = 0
    pdr_issues: int = 0


STEP_LAYOUT = {
    "step1": {
        "dir_arg": "step1_dir",
        "flat_files": ("raw_metrics.csv", "aggregated_results.csv"),
        "nested_files": ("raw_metrics.csv", "aggregated_results.csv"),
    },
    "step2": {
        "dir_arg": "step2_dir",
        "flat_files": ("raw_results.csv", "aggregated_results.csv"),
        "nested_files": ("raw_results.csv", "aggregated_results.csv"),
    },
}


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


def _read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _parse_sized_rep_path(path: Path) -> tuple[str, int, int] | None:
    match = re.search(r"size_(\d+)/rep_(\d+)/(.*)$", path.as_posix())
    if not match:
        return None
    return match.group(3), int(match.group(1)), int(match.group(2))


def _parse_int_csv_list(values: str | None) -> list[int] | None:
    if not values:
        return None
    parsed: list[int] = []
    for item in values.split(","):
        text = item.strip()
        if not text:
            continue
        parsed.append(int(text))
    return parsed or None


def _collect_nested_integrity(
    *,
    step_label: str,
    step_dir: Path,
    nested_files: tuple[str, ...],
    expected_sizes: list[int] | None,
    expected_replications: int | None,
) -> tuple[list[list[str]], list[list[str]], IntegrityIssueCounter]:
    report_rows: list[list[str]] = []
    console_rows: list[list[str]] = []
    issues = IntegrityIssueCounter()
    discovered: dict[tuple[int, int], set[str]] = {}
    existing_headers: dict[str, dict[tuple[str, ...], list[str]]] = {
        csv_name: {} for csv_name in nested_files
    }

    for path in sorted(step_dir.glob("size_*/rep_*/*.csv")):
        parsed = _parse_sized_rep_path(path)
        if parsed is None:
            continue
        filename, size, rep = parsed
        if filename not in nested_files:
            continue
        discovered.setdefault((size, rep), set()).add(filename)
        header = tuple(_read_header(path))
        existing_headers[filename].setdefault(header, []).append(str(path))

    discovered_sizes = sorted({size for size, _ in discovered})
    discovered_reps = sorted({rep for _, rep in discovered})
    target_sizes = expected_sizes or discovered_sizes
    if expected_replications is None:
        target_reps = discovered_reps
    else:
        target_reps = list(range(1, expected_replications + 1))

    if not target_sizes or not target_reps:
        console_rows.append(
            [
                f"{step_label}:nested",
                "n/a",
                "n/a",
                "n/a",
                "0",
                "-",
                "absence de sous-dossiers size_*/rep_* exploitables",
            ]
        )
        report_rows.append(
            [
                str(step_dir),
                "nested_scan",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "0",
                "-",
                "-",
                "0",
                "0",
                "Aucune donnée imbriquée détectée",
            ]
        )
        return report_rows, console_rows, issues

    for size in target_sizes:
        missing_reps: list[int] = []
        for rep in target_reps:
            present = discovered.get((size, rep), set())
            missing_files = sorted(set(nested_files) - present)
            if missing_files:
                issues.missing_files += len(missing_files)
                if len(missing_files) == len(nested_files):
                    missing_reps.append(rep)
                report_rows.append(
                    [
                        str(step_dir),
                        "nested_presence",
                        str(size),
                        str(rep),
                        "n/a",
                        "n/a",
                        "n/a",
                        "0",
                        "-",
                        "-",
                        "0",
                        "0",
                        ";".join(missing_files),
                    ]
                )
        if missing_reps:
            issues.replication_holes += len(missing_reps)
            console_rows.append(
                [
                    f"{step_label}:nested",
                    "n/a",
                    "n/a",
                    f"size_{size}",
                    "0",
                    _format_list({str(rep) for rep in missing_reps}),
                    "réplications manquantes",
                ]
            )

    for csv_name in nested_files:
        variants = existing_headers[csv_name]
        if len(variants) <= 1:
            continue
        issues.column_inconsistencies += len(variants) - 1
        for idx, (header, paths) in enumerate(sorted(variants.items(), key=lambda item: len(item[1]), reverse=True), start=1):
            report_rows.append(
                [
                    str(step_dir),
                    "column_schema",
                    "*",
                    "*",
                    csv_name,
                    "schema",
                    "*",
                    "0",
                    "-",
                    "-",
                    "0",
                    "0",
                    f"variant={idx};cols={';'.join(header)};sample={paths[0]}",
                ]
            )
        console_rows.append(
            [
                f"{step_label}:nested",
                csv_name,
                "schema",
                "*",
                "0",
                "-",
                f"incohérence colonnes ({len(variants)} variantes)",
            ]
        )

    return report_rows, console_rows, issues


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
    parser.add_argument(
        "--network-sizes",
        type=str,
        default=None,
        help="Tailles attendues pour la validation imbriquée (ex: 50,100,150).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Nombre attendu de réplications (rep_1..rep_N).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retourne un code non nul si des anomalies d'intégrité sont détectées.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_sizes = _parse_int_csv_list(args.network_sizes)
    files = [
        args.step1_dir / "raw_metrics.csv",
        args.step1_dir / "aggregates" / "aggregated_results.csv",
        args.step2_dir / "raw_results.csv",
        args.step2_dir / "aggregates" / "aggregated_results.csv",
    ]
    report_rows: list[list[str]] = []
    console_rows: list[list[str]] = []
    global_issues = IntegrityIssueCounter()
    for path in files:
        rows = _read_csv(path)
        groups, all_sizes = _collect_stats(rows, args.tolerance)
        if not rows:
            report_rows.append(
                [
                    str(path),
                    "flat_aggregates",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "0",
                    "-",
                    "-",
                    "0",
                    "0",
                    "fichier absent ou vide",
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
                    "fichier absent",
                ]
            )
            global_issues.missing_files += 1
            continue
        for stats in sorted(
            groups.values(),
            key=lambda item: (item.algo, item.snir_mode, item.cluster),
        ):
            missing_sizes = all_sizes - stats.sizes
            report_rows.append(
                [
                    str(path),
                    "flat_aggregates",
                    "*",
                    "*",
                    stats.algo,
                    stats.snir_mode,
                    stats.cluster,
                    str(stats.row_count),
                    _format_list(stats.sizes),
                    _format_list(missing_sizes),
                    str(stats.pdr_issues),
                    str(stats.pdr_checked),
                    "",
                ]
            )
            global_issues.pdr_issues += stats.pdr_issues
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

    for step_name, config in STEP_LAYOUT.items():
        step_dir = getattr(args, config["dir_arg"])
        nested_report_rows, nested_console_rows, nested_issues = _collect_nested_integrity(
            step_label=step_name,
            step_dir=step_dir,
            nested_files=config["nested_files"],
            expected_sizes=expected_sizes,
            expected_replications=args.replications,
        )
        report_rows.extend(nested_report_rows)
        console_rows.extend(nested_console_rows)
        global_issues.missing_files += nested_issues.missing_files
        global_issues.replication_holes += nested_issues.replication_holes
        global_issues.column_inconsistencies += nested_issues.column_inconsistencies

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "source_file",
        "scope",
        "network_size",
        "replication",
        "algo",
        "snir_mode",
        "cluster",
        "row_count",
        "sizes_present",
        "missing_sizes",
        "pdr_issues",
        "pdr_checked",
        "details",
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
    print(
        "Résumé anomalies: "
        f"missing_files={global_issues.missing_files}, "
        f"replication_holes={global_issues.replication_holes}, "
        f"column_inconsistencies={global_issues.column_inconsistencies}, "
        f"pdr_issues={global_issues.pdr_issues}"
    )
    has_integrity_issues = any(
        (
            global_issues.missing_files,
            global_issues.replication_holes,
            global_issues.column_inconsistencies,
            global_issues.pdr_issues,
        )
    )
    if args.strict and has_integrity_issues:
        print("Mode strict: anomalies détectées, code retour=1.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
