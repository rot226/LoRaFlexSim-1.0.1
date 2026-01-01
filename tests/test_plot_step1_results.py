from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import plot_step1_results


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_missing_snir_state_excluded_from_mixed(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1"
    results_dir.mkdir()
    csv_path = results_dir / "results.csv"

    _write_csv(
        csv_path,
        [
            {
                "algorithm": "adr",
                "num_nodes": "10",
                "packet_interval_s": "1",
                "PDR": "0.9",
                "DER": "0.8",
            }
        ],
    )

    with pytest.warns(RuntimeWarning, match="Aucun état SNIR explicite"):
        records = plot_step1_results._load_step1_records(results_dir)

    assert len(records) == 1
    record = records[0]
    assert record["snir_state"] is None
    assert record["snir_detected"] is False
    assert plot_step1_results._record_matches_state(record, "snir_unknown") is False
