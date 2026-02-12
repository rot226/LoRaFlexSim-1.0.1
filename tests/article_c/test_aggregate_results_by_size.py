from __future__ import annotations

import csv
from pathlib import Path

from article_c.common.csv_io import aggregate_results_by_size


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_writes_per_size_aggregation_with_audit_field(tmp_path: Path) -> None:
    results_dir = tmp_path / "step1" / "results"
    _write_csv(
        results_dir / "by_size" / "size_50" / "rep_1" / "aggregated_results.csv",
        [{"network_size": "50", "algo": "adr", "snir_mode": "snir_on", "pdr_mean": "0.9"}],
    )
    _write_csv(
        results_dir / "by_size" / "size_50" / "rep_2" / "aggregated_results.csv",
        [{"network_size": "50", "algo": "adr", "snir_mode": "snir_off", "pdr_mean": "0.7"}],
    )

    stats = aggregate_results_by_size(results_dir)

    assert stats["size_count"] == 1
    assert stats["size_row_count"] == 2
    size_rows = _read_csv(results_dir / "by_size" / "size_50" / "aggregated_results.csv")
    assert len(size_rows) == 2
    assert {row["source_size_dir"] for row in size_rows} == {"size_50"}


def test_optionally_writes_global_aggregation(tmp_path: Path) -> None:
    results_dir = tmp_path / "step2" / "results"
    _write_csv(
        results_dir / "by_size" / "size_20" / "rep_1" / "aggregated_results.csv",
        [{"network_size": "20", "algo": "adr", "snir_mode": "snir_on", "reward_mean": "0.6"}],
    )
    _write_csv(
        results_dir / "by_size" / "size_40" / "rep_1" / "aggregated_results.csv",
        [{"network_size": "40", "algo": "adr", "snir_mode": "snir_on", "reward_mean": "0.5"}],
    )

    stats = aggregate_results_by_size(results_dir, write_global_aggregated=True)

    assert stats["size_count"] == 2
    assert stats["global_row_count"] == 2
    global_rows = _read_csv(results_dir / "aggregated_results.csv")
    assert len(global_rows) == 2
    assert {row["source_size_dir"] for row in global_rows} == {"size_20", "size_40"}
