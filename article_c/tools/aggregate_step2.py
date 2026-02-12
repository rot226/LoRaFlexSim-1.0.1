"""Agrège les résultats Step2 depuis by_size/size_*/rep_* vers results/."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from article_c.common.config import BASE_DIR
from article_c.common.csv_io import STEP2_EXPECTED_METRICS, aggregate_results, atomic_write_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agrège Step2 sans exécuter de simulation."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE_DIR / "step2" / "results",
        help="Dossier des résultats Step2 (défaut: article_c/step2/results).",
    )
    return parser


def _coerce(value: str, key: str) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if key in {"network_size", "replication", "round", "seed", "node_id", "packet_id", "sf"}:
        if number.is_integer():
            return int(number)
    return number


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _coerce(value, key) for key, value in row.items()})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = list(rows[0].keys()) if rows else ["network_size", "algo", "snir_mode", "cluster"]
    atomic_write_csv(
        path,
        header,
        [[row.get(key, "") for key in header] for row in rows],
    )


def main() -> int:
    args = _build_parser().parse_args()
    results_dir = args.results_dir.resolve()
    by_size_dir = results_dir / "by_size"
    raw_paths = sorted(by_size_dir.glob("size_*/rep_*/raw_results.csv"))
    if not raw_paths:
        print(f"Aucun raw_results.csv détecté sous {by_size_dir / 'size_<N>/rep_<R>'}.")
        return 1

    raw_rows: list[dict[str, object]] = []
    for raw_path in raw_paths:
        raw_rows.extend(_load_rows(raw_path))

    aggregated_rows, intermediate_rows = aggregate_results(
        raw_rows,
        expected_metrics=STEP2_EXPECTED_METRICS,
        step_label="Step2",
    )

    _write_csv(results_dir / "aggregates" / "aggregated_results.csv", aggregated_rows)
    _write_csv(results_dir / "aggregates" / "aggregated_results_by_size.csv", aggregated_rows)
    _write_csv(results_dir / "aggregates" / "aggregated_results_by_replication.csv", intermediate_rows)

    print(
        "Agrégation Step2 terminée: aggregated_results.csv, "
        "aggregated_results_by_size.csv, aggregated_results_by_replication.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
