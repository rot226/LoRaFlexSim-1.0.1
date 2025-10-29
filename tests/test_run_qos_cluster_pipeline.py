from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_qos_cluster_pipeline


@pytest.fixture
def tmp_dirs(tmp_path: Path):
    results = tmp_path / "res"
    figures = tmp_path / "fig"
    return results, figures


def test_pipeline_runs_runner_and_plotter(tmp_dirs):
    results_dir, figures_dir = tmp_dirs
    calls = []

    def fake_runner(**kwargs):
        calls.append(("runner", kwargs))
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        return {"summary_path": "results.json", "report_path": "report.md"}

    def fake_plotter(results, figures):
        calls.append(("plotter", results, figures))
        return True

    summary = run_qos_cluster_pipeline.main(
        [
            "--preset",
            "quick",
            "--quiet",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
        ],
        runner=fake_runner,
        plotter=fake_plotter,
    )
    assert summary["summary_path"] == "results.json"
    assert ("runner",) == calls[0][:1]
    assert calls[1][0] == "plotter"
    assert Path(calls[1][1]) == results_dir
    assert Path(calls[1][2]) == figures_dir


def test_pipeline_skip_plots(tmp_dirs):
    results_dir, figures_dir = tmp_dirs
    calls = []

    def fake_runner(**kwargs):
        calls.append("runner")
        return {}

    summary = run_qos_cluster_pipeline.main(
        [
            "--preset",
            "quick",
            "--skip-plots",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
            "--quiet",
        ],
        runner=fake_runner,
        plotter=lambda *_: (_ for _ in ()).throw(AssertionError("plotter ne doit pas être appelé")),
    )
    assert summary == {}
    assert calls == ["runner"]
