from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.aggregate_step1_results import aggregate_step1_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: set[str] = set()
    for row in rows:
        fieldnames.update(row.keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def test_rejects_missing_snir_state(tmp_path: Path) -> None:
    csv_path = tmp_path / "snir_on" / "algo_snir-on.csv"
    _write_csv(
        csv_path,
        [
            {
                "algorithm": "algo",
                "num_nodes": "10",
                "packet_interval_s": "1.0",
                "PDR": "0.9",
            }
        ],
    )

    with pytest.raises(ValueError, match="snir_state est manquant"):
        aggregate_step1_results(tmp_path, strict_snir_detection=False, split_snir=False)


def test_rejects_incoherent_snir_state(tmp_path: Path) -> None:
    csv_path = tmp_path / "snir_off" / "algo_snir-off.csv"
    _write_csv(
        csv_path,
        [
            {
                "algorithm": "algo",
                "num_nodes": "10",
                "packet_interval_s": "1.0",
                "PDR": "0.9",
                "snir_state": "snir_on",
                "with_snir": "true",
            }
        ],
    )

    with pytest.raises(ValueError, match="suffixe implique snir_off"):
        aggregate_step1_results(tmp_path, strict_snir_detection=False, split_snir=False)
