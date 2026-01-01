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


def test_official_run_outputs_only_extended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results" / "step1"
    results_dir.mkdir(parents=True)
    figures_dir = tmp_path / "figures"

    _write_csv(
        results_dir / "summary.csv",
        [
            {
                "algorithm": "adr",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR_mean": "0.9",
            }
        ],
    )
    _write_csv(
        results_dir / "raw_index.csv",
        [
            {
                "algorithm": "adr",
                "snir_state": "snir_on",
                "packet_interval_s": "60",
                "DER": "0.1",
            }
        ],
    )

    def _fake_plot_summary_bars(records: list[dict[str, object]], figures_path: Path) -> None:
        figures_path.mkdir(parents=True, exist_ok=True)
        (figures_path / "summary.png").write_text("summary")

    def _fake_plot_cdf(records: list[dict[str, object]], figures_path: Path) -> None:
        figures_path.mkdir(parents=True, exist_ok=True)
        (figures_path / "cdf.png").write_text("cdf")

    def _fake_plot_snir_comparison(records: list[dict[str, object]], figures_path: Path) -> None:
        figures_path.mkdir(parents=True, exist_ok=True)
        (figures_path / "compare.png").write_text("compare")

    monkeypatch.setattr(plot_step1_results, "_plot_summary_bars", _fake_plot_summary_bars)
    monkeypatch.setattr(plot_step1_results, "_plot_cdf", _fake_plot_cdf)
    monkeypatch.setattr(plot_step1_results, "_plot_snir_comparison", _fake_plot_snir_comparison)
    monkeypatch.setattr(plot_step1_results, "_apply_ieee_style", lambda: None)
    monkeypatch.setattr(plot_step1_results, "plt", object())

    plot_step1_results.generate_step1_figures(
        results_dir,
        figures_dir,
        use_summary=True,
        plot_cdf=True,
        compare_snir=True,
        official=True,
    )

    generated = list(figures_dir.rglob("*.png"))
    assert generated, "Aucune figure officielle n'a été générée."
    assert all("extended" in path.parts for path in generated)
    assert not list((figures_dir / "step1").glob("*.png"))
