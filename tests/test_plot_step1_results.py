from __future__ import annotations

import csv
import warnings
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


def test_warns_when_snir_unknown_detected(tmp_path: Path) -> None:
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
                "snir_state": "snir_unknown",
                "PDR": "0.9",
                "DER": "0.8",
            }
        ],
    )

    with pytest.warns(RuntimeWarning, match="snir_unknown"):
        records = plot_step1_results._load_step1_records(results_dir)

    assert records, "Les enregistrements SNIR inconnus doivent être chargés."


def test_missing_snir_mean_raises_for_snir_on(tmp_path: Path) -> None:
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
                "snir_state": "snir_on",
                "PDR": "0.9",
                "DER": "0.8",
            }
        ],
    )

    with pytest.raises(ValueError, match="snir_mean"):
        plot_step1_results._load_step1_records(results_dir)


def test_missing_snir_mean_raises_for_use_snir_true(tmp_path: Path) -> None:
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
                "use_snir": "true",
                "PDR": "0.9",
                "DER": "0.8",
            }
        ],
    )

    with pytest.raises(ValueError, match="snir_mean"):
        plot_step1_results._load_step1_records(results_dir)


def test_mixed_plot_filters_snir_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[tuple[str, list[str]]] = []

    class _FakeLine:
        def set_linewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markersize(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markeredgewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[_FakeLine] = []

        def plot(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def errorbar(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[_FakeLine], list[str]]:
            return self._lines, self._labels

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, output: Path, **_kwargs: object) -> None:
            captures.append((str(output), list(self._axis._labels)))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, _fig: _FakeFigure) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    records = [
        {
            "algorithm": "algo",
            "num_nodes": 10,
            "packet_interval_s": 1.0,
            "PDR": 0.9,
            "snir_state": "snir_on",
            "snir_detected": True,
        },
        {
            "algorithm": "algo",
            "num_nodes": 20,
            "packet_interval_s": 1.0,
            "PDR": 0.85,
            "snir_state": "snir_off",
            "snir_detected": True,
        },
        {
            "algorithm": "algo",
            "num_nodes": 30,
            "packet_interval_s": 1.0,
            "PDR": 0.5,
            "snir_state": "snir_unknown",
            "snir_detected": True,
        },
    ]

    plot_step1_results._plot_global_metric(
        records,
        "PDR",
        "PDR global",
        "pdr_global",
        tmp_path,
    )

    mixed_labels = [
        labels for path, labels in captures if "_snir-mixed" in path
    ]
    assert mixed_labels, "Aucun tracé mixte n'a été généré."
    assert all(
        "SNIR inconnu" not in label for labels in mixed_labels for label in labels
    )


def test_compare_mixed_plot_excludes_unknown_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures: list[tuple[str, list[str]]] = []

    class _FakeLine:
        def set_linewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markersize(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markeredgewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[_FakeLine] = []

        def plot(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def errorbar(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[_FakeLine], list[str]]:
            return self._lines, self._labels

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, output: Path, **_kwargs: object) -> None:
            captures.append((str(output), list(self._axis._labels)))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, _fig: _FakeFigure) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    records = [
        {
            "algorithm": "algo",
            "num_nodes": 10,
            "packet_interval_s": 1.0,
            "PDR": 0.9,
            "snir_state": "snir_on",
            "snir_detected": True,
        },
        {
            "algorithm": "algo",
            "num_nodes": 20,
            "packet_interval_s": 1.0,
            "PDR": 0.85,
            "snir_state": "snir_off",
            "snir_detected": True,
        },
        {
            "algorithm": "algo",
            "num_nodes": 30,
            "packet_interval_s": 1.0,
            "PDR": 0.5,
            "snir_state": "snir_unknown",
            "snir_detected": True,
        },
    ]

    plot_step1_results._plot_snir_comparison(records, tmp_path)

    mixed_labels = [
        labels for path, labels in captures if "_snir-mixed" in path
    ]
    assert mixed_labels, "Aucun tracé mixte n'a été généré."
    assert all(
        "SNIR inconnu" not in label for labels in mixed_labels for label in labels
    )


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

    def _fake_plot_summary_bars(
        records: list[dict[str, object]],
        figures_path: Path,
        forced_algorithm: str | None = None,
    ) -> None:
        figures_path.mkdir(parents=True, exist_ok=True)
        (figures_path / "summary.png").write_text("summary")

    def _fake_plot_cdf(
        records: list[dict[str, object]],
        figures_path: Path,
        forced_algorithm: str | None = None,
    ) -> None:
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
        official_only=True,
    )

    generated = list(figures_dir.rglob("*.png"))
    assert generated, "Aucune figure officielle n'a été générée."
    assert all("extended" in path.parts for path in generated)
    assert not list((figures_dir / "step1").glob("*.png"))


def test_summary_csv_does_not_exclude_mixra_opt(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "step1"
    results_dir.mkdir(parents=True)
    _write_csv(
        results_dir / "summary.csv",
        [
            {
                "algorithm": "MixRA-Opt",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR_mean": "0.9",
            }
        ],
    )
    _write_csv(
        results_dir / "summary_snir_on.csv",
        [
            {
                "algorithm": "adr",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR_mean": "0.8",
            }
        ],
    )

    records = plot_step1_results._load_comparison_records(
        results_dir, use_summary=True, strict=False
    )

    assert any(record.get("algorithm") == "mixra_opt" for record in records)


def test_extended_summary_includes_mixra_opt_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = tmp_path / "summary.csv"
    _write_csv(
        summary_path,
        [
            {
                "algorithm": "MixRA-Opt",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR_mean": "0.9",
                "PDR_std": "0.01",
            }
        ],
    )

    records = plot_step1_results._load_summary_records(summary_path)
    assert any(record.get("algorithm") == "mixra_opt" for record in records)

    captured_labels: list[str] = []

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []

        def bar(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_xticks(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_xticklabels(self, labels: list[str], *_args: object, **_kwargs: object) -> None:
            captured_labels.extend(labels)

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[object], list[str]]:
            return [object()], ["snir"]

    class _FakeFigure:
        def tight_layout(self) -> None:
            return None

        def savefig(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            return _FakeFigure(), _FakeAxis()

        def close(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    plot_step1_results._plot_summary_bars(records, tmp_path)

    assert any("mixra_opt" in label for label in captured_labels)


def test_extended_cdf_legend_includes_mixra_opt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures: list[list[str]] = []

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[object] = []

        def step(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(object())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[object], list[str]]:
            return self._lines, self._labels

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, *_args: object, **_kwargs: object) -> None:
            captures.append(list(self._axis._labels))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    records = [
        {
            "algorithm": "mixra_opt",
            "snir_state": "snir_on",
            "snir_detected": True,
            "DER": 0.1,
        },
        {
            "algorithm": "adr",
            "snir_state": "snir_on",
            "snir_detected": True,
            "DER": 0.2,
        },
    ]

    plot_step1_results._plot_cdf(
        records,
        tmp_path,
        forced_algorithm="mixra_opt",
    )

    assert any(
        "mixra_opt" in label for labels in captures for label in labels
    ), "mixra_opt absent de la légende CDF étendue."


def test_cli_includes_mixra_opt_in_standard_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "step1"
    results_dir.mkdir()
    csv_path = results_dir / "results.csv"

    _write_csv(
        csv_path,
        [
            {
                "algorithm": "Opt",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR": "0.92",
                "DER": "0.88",
                "snir_mean": "5.0",
            },
            {
                "algorithm": "Opt",
                "snir_state": "snir_off",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR": "0.85",
                "DER": "0.8",
                "snr_mean": "4.0",
            },
        ],
    )

    captures: list[tuple[str, list[str]]] = []

    class _FakeLine:
        def set_linewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markersize(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markeredgewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[_FakeLine] = []

        def plot(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def errorbar(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[_FakeLine], list[str]]:
            return self._lines, self._labels

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, output: Path, **_kwargs: object) -> None:
            captures.append((str(output), list(self._axis._labels)))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, _fig: _FakeFigure) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_step1_results, "_apply_ieee_style", lambda: None)
    monkeypatch.setattr(plot_step1_results, "plot_distribution_by_state", lambda *_args, **_kwargs: None)

    plot_step1_results.main(
        [
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(tmp_path / "figures"),
        ]
    )

    assert any(
        "mixra_opt" in label for _, labels in captures for label in labels
    ), "mixra_opt absent des labels des figures standard."
    assert any(
        "algo_mixra_opt" in path for path, _ in captures
    ), "Aucun fichier SNIR compare pour mixra_opt."


def test_cli_includes_mixra_opt_in_all_global_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "step1"
    results_dir.mkdir()
    csv_path = results_dir / "results.csv"

    _write_csv(
        csv_path,
        [
            {
                "algorithm": "Opt",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR": "0.92",
                "DER": "0.88",
                "snir_mean": "5.0",
                "collisions": "2",
                "collisions_snir": "1",
            },
            {
                "algorithm": "Opt",
                "snir_state": "snir_off",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "PDR": "0.85",
                "DER": "0.8",
                "snr_mean": "4.0",
                "collisions": "4",
                "collisions_snir": "3",
            },
        ],
    )

    captures: list[tuple[str, list[str]]] = []

    class _FakeLine:
        def set_linewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markersize(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markeredgewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[_FakeLine] = []

        def plot(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def errorbar(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[_FakeLine], list[str]]:
            return self._lines, self._labels

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, output: Path, **_kwargs: object) -> None:
            captures.append((str(output), list(self._axis._labels)))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, _fig: _FakeFigure) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_step1_results, "_apply_ieee_style", lambda: None)
    monkeypatch.setattr(plot_step1_results, "_plot_cluster_der", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_step1_results, "_plot_cluster_pdr", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_step1_results, "plot_distribution_by_state", lambda *_args, **_kwargs: None)

    plot_step1_results.main(
        [
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(tmp_path / "figures"),
        ]
    )

    expected_prefixes = [
        "step1_pdr_global",
        "step1_der_global",
        "step1_collisions",
        "step1_collisions_snir",
        "step1_snir_mean",
    ]

    for prefix in expected_prefixes:
        matching = [
            labels
            for path, labels in captures
            if prefix in path
        ]
        assert matching, f"Aucune sortie générée pour {prefix}."
        assert any(
            "mixra_opt" in label for labels in matching for label in labels
        ), f"mixra_opt absent des labels pour {prefix}."


def test_plot_trajectories_includes_mixra_opt_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "step1"
    results_dir.mkdir()
    csv_path = results_dir / "results.csv"

    _write_csv(
        csv_path,
        [
            {
                "algorithm": "Opt",
                "snir_state": "snir_on",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "random_seed": "1",
                "PDR": "0.9",
                "DER": "0.8",
                "snir_mean": "5.0",
            },
            {
                "algorithm": "Opt",
                "snir_state": "snir_off",
                "num_nodes": "10",
                "packet_interval_s": "60",
                "random_seed": "1",
                "PDR": "0.85",
                "DER": "0.75",
                "snr_mean": "4.0",
            },
            {
                "algorithm": "Opt",
                "snir_state": "snir_on",
                "num_nodes": "20",
                "packet_interval_s": "60",
                "random_seed": "1",
                "PDR": "0.88",
                "DER": "0.78",
                "snir_mean": "5.5",
            },
            {
                "algorithm": "Opt",
                "snir_state": "snir_off",
                "num_nodes": "20",
                "packet_interval_s": "60",
                "random_seed": "1",
                "PDR": "0.82",
                "DER": "0.7",
                "snr_mean": "3.5",
            },
        ],
    )

    records = plot_step1_results._load_step1_records(results_dir)
    assert any(record.get("algorithm") == "mixra_opt" for record in records)

    captures: list[tuple[str, list[str]]] = []

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[object] = []

        def plot(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(object())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[object], list[str]]:
            return self._lines, self._labels

        def get_lines(self) -> list[object]:
            return self._lines

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, output: Path, **_kwargs: object) -> None:
            captures.append((str(output), list(self._axis._labels)))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_cmap(self, *_args: object, **_kwargs: object) -> object:
            class _Map:
                N = 10

                def __call__(self, idx: int) -> tuple[float, float, float, float]:
                    return (0.0, 0.0, 0.0, 1.0)

            return _Map()

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    plot_step1_results._plot_trajectories(records, tmp_path)

    assert any("mixra_opt" in path for path, _ in captures)
    assert any(
        "SNIR activé" in label for _, labels in captures for label in labels
    )
    assert any(
        "SNIR désactivé" in label for _, labels in captures for label in labels
    )


def test_plot_trajectories_has_seed_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "step1"
    results_dir.mkdir()
    csv_path = results_dir / "results.csv"

    rows = []
    for seed in (1, 2):
        for nodes in (10, 20):
            rows.append(
                {
                    "algorithm": "Opt",
                    "snir_state": "snir_on",
                    "num_nodes": str(nodes),
                    "packet_interval_s": "60",
                    "random_seed": str(seed),
                    "PDR": "0.9",
                    "DER": "0.8",
                    "snir_mean": "5.0",
                }
            )
            rows.append(
                {
                    "algorithm": "Opt",
                    "snir_state": "snir_off",
                    "num_nodes": str(nodes),
                    "packet_interval_s": "60",
                    "random_seed": str(seed),
                    "PDR": "0.85",
                    "DER": "0.75",
                    "snr_mean": "4.0",
                }
            )

    _write_csv(csv_path, rows)
    records = plot_step1_results._load_step1_records(results_dir)

    series_lengths: dict[str, list[int]] = {}

    class _FakeAxis:
        def __init__(self) -> None:
            self._lines: list[object] = []

        def plot(self, xs: list[float], *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                series_lengths.setdefault(label, []).append(len(xs))
            self._lines.append(object())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[object], list[str]]:
            return self._lines, list(series_lengths)

        def get_lines(self) -> list[object]:
            return self._lines

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_cmap(self, *_args: object, **_kwargs: object) -> object:
            class _Map:
                N = 10

                def __call__(self, idx: int) -> tuple[float, float, float, float]:
                    return (0.0, 0.0, 0.0, 1.0)

            return _Map()

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    plot_step1_results._plot_trajectories(records, tmp_path)

    for seed in (1, 2):
        for state_label in ("SNIR activé", "SNIR désactivé"):
            label = f"seed {seed} – {state_label}"
            assert label in series_lengths
            assert any(length >= 2 for length in series_lengths[label])


def test_plot_distribution_by_state_has_no_deprecation_warning(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    if plot_step1_results.plt is None:
        pytest.skip("Matplotlib indisponible dans cet environnement.")

    records = [
        {
            "snir_state": "snir_on",
            "snir_detected": True,
            "snir_mean": 5.0,
            "DER": 0.8,
            "collisions": 1,
        },
        {
            "snir_state": "snir_off",
            "snir_detected": True,
            "snir_mean": 3.5,
            "DER": 0.7,
            "collisions": 2,
        },
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        plot_step1_results.plot_distribution_by_state(records, tmp_path)

    deprecations = [
        warning for warning in caught if issubclass(warning.category, DeprecationWarning)
    ]
    assert not deprecations, "Aucun warning de dépréciation ne doit être émis."


def test_global_metric_deduplicates_algorithm_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures: list[tuple[str, list[str]]] = []

    class _FakeLine:
        def set_linewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markersize(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_markeredgewidth(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _FakeAxis:
        def __init__(self) -> None:
            self._labels: list[str] = []
            self._lines: list[_FakeLine] = []

        def plot(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def errorbar(self, *_args: object, label: str | None = None, **_kwargs: object) -> None:
            if label:
                self._labels.append(label)
            self._lines.append(_FakeLine())

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def get_legend_handles_labels(self) -> tuple[list[_FakeLine], list[str]]:
            return self._lines, self._labels

    class _FakeFigure:
        def __init__(self, axis: _FakeAxis) -> None:
            self._axis = axis

        def tight_layout(self) -> None:
            return None

        def savefig(self, output: Path, **_kwargs: object) -> None:
            captures.append((str(output), list(self._axis._labels)))

    class _FakePlt:
        def subplots(self, **_kwargs: object) -> tuple[_FakeFigure, _FakeAxis]:
            axis = _FakeAxis()
            return _FakeFigure(axis), axis

        def close(self, _fig: _FakeFigure) -> None:
            return None

    monkeypatch.setattr(plot_step1_results, "plt", _FakePlt())
    monkeypatch.setattr(plot_step1_results, "_format_axes", lambda *_args, **_kwargs: None)

    records = [
        {
            "algorithm": "dup",
            "num_nodes": 10,
            "packet_interval_s": 1.0,
            "PDR": 0.9,
            "snir_state": "snir_on",
            "snir_detected": True,
        },
        {
            "algorithm": "dup",
            "num_nodes": 20,
            "packet_interval_s": 1.0,
            "PDR": 0.85,
            "snir_state": "snir_on",
            "snir_detected": True,
        },
        {
            "algorithm": "autre",
            "num_nodes": 10,
            "packet_interval_s": 1.0,
            "PDR": 0.8,
            "snir_state": "snir_on",
            "snir_detected": True,
        },
    ]

    plot_step1_results._plot_global_metric(
        records,
        "PDR",
        "PDR global",
        "pdr_global",
        tmp_path,
    )

    assert captures, "Aucune figure n'a été générée."
    for _path, labels in captures:
        assert len(labels) == len(set(labels)), "Les algorithmes doivent être uniques par figure."
