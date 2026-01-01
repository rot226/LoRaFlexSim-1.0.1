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

    assert records == []


def test_mixed_variants_exclude_snir_unknown() -> None:
    seen_states: list[list[str]] = []

    def render(states: list[str], suffix: str, title: str) -> None:
        if suffix == "_snir-mixed":
            seen_states.append(states)

    plot_step1_results._render_snir_variants(
        render,
        on_title="SNIR activé",
        off_title="SNIR désactivé",
        mixed_title="SNIR mixte",
    )

    assert seen_states == [["snir_on", "snir_off"]]
