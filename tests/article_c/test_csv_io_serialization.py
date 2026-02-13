from __future__ import annotations

import csv
from pathlib import Path

import pytest

from article_c.common.csv_io import write_simulation_results


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    return rows[0]


def test_write_simulation_results_normalizes_network_size_and_density_types(tmp_path: Path) -> None:
    raw_rows = [
        {
            "network_size": "10.0",
            "density": "10",
            "algo": "ucb1",
            "snir_mode": "snir_on",
            "cluster": "all",
            "reward": 0.5,
            "success_rate": 0.8,
        }
    ]

    write_simulation_results(tmp_path, raw_rows)

    raw_row = _read_first_row(tmp_path / "raw_results.csv")
    aggregated_row = _read_first_row(tmp_path / "aggregated_results.csv")

    assert raw_row["network_size"] == "10"
    assert raw_row["density"] == "10.0"
    assert aggregated_row["network_size"] == "10"
    assert "density" not in aggregated_row


@pytest.mark.parametrize("invalid_size", [None, "", -1, "-2", float("nan"), float("inf")])
def test_write_simulation_results_rejects_invalid_network_size(
    tmp_path: Path,
    invalid_size: object,
) -> None:
    raw_rows = [
        {
            "network_size": invalid_size,
            "algo": "ucb1",
            "snir_mode": "snir_on",
            "cluster": "all",
            "reward": 0.5,
            "success_rate": 0.8,
        }
    ]

    with pytest.raises((AssertionError, ValueError)):
        write_simulation_results(tmp_path, raw_rows)


@pytest.mark.parametrize("invalid_density", ["nan", "inf", -0.1])
def test_write_simulation_results_rejects_invalid_density(
    tmp_path: Path,
    invalid_density: object,
) -> None:
    raw_rows = [
        {
            "network_size": 10,
            "density": invalid_density,
            "algo": "ucb1",
            "snir_mode": "snir_on",
            "cluster": "all",
            "reward": 0.5,
            "success_rate": 0.8,
        }
    ]

    with pytest.raises(ValueError):
        write_simulation_results(tmp_path, raw_rows)
