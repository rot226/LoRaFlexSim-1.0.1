import pathlib
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mobilesfrdth import cli


def test_main_rejects_unsupported_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 10, 9, "final", 0))

    exit_code = cli.main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Version Python non supportée" in captured.err
    assert "utiliser une version >=3.11 et <3.15" in captured.err


def test_main_help_supported_python(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 8, "final", 0))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0


def test_build_parser_accepts_aggregate_acceleration_flags(tmp_path):
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "aggregate",
            "--results",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--summary-only",
            "--skip-sinr-cdf",
            "--skip-sf-distribution",
        ]
    )

    assert args.summary_only is True
    assert args.skip_sinr_cdf is True
    assert args.skip_sf_distribution is True


def test_build_parser_accepts_run_profile_without_grid(tmp_path):
    parser = cli.build_parser()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("demo: true\n", encoding="utf-8")

    args = parser.parse_args(
        [
            "run",
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "out"),
            "--profile",
            "smoke",
        ]
    )

    assert args.profile == "smoke"
    assert args.grid is None


def test_cmd_run_uses_profile_and_prints_progress(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("demo: true\n", encoding="utf-8")

    observed = {}

    class _FakeOrchestrator:
        def __init__(self, *, output_root):
            observed["output_root"] = output_root

        def execute_jobs(self, jobs, **kwargs):
            observed["jobs"] = jobs
            callback = kwargs["on_run_complete"]
            callback(
                SimpleNamespace(run_id="run_01", success=True),
                1,
                2,
                1,
                0,
                45.0,
            )
            callback(
                SimpleNamespace(run_id="run_02", success=False),
                2,
                2,
                1,
                1,
                None,
            )
            return SimpleNamespace(
                reports=[
                    SimpleNamespace(run_id="run_01", success=True, run_dir=tmp_path / "run_01", error=None),
                    SimpleNamespace(run_id="run_02", success=False, run_dir=tmp_path / "run_02", error="boom"),
                ],
                total_jobs=2,
                skipped_runs=0,
                scheduled_runs=2,
                failed_reports=[SimpleNamespace(run_id="run_02", run_dir=tmp_path / "run_02", error="boom")],
                interrupted=False,
            )

    monkeypatch.setattr("mobilesfrdth.cli.GridRunOrchestrator", _FakeOrchestrator, raising=False)
    monkeypatch.setattr("mobilesfrdth.simulator.engine.GridRunOrchestrator", _FakeOrchestrator)

    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "out"),
            "--profile",
            "smoke",
        ]
    )

    exit_code = cli.cmd_run(args)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Profil sélectionné: smoke" in out
    assert "[1/2] run_01: succès | ETA=00:00:45" in out
    assert "[2/2] run_02: échec | ETA=N/A" in out
    assert len(observed["jobs"]) > 0


def test_cmd_run_returns_130_when_interrupted(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("demo: true\n", encoding="utf-8")

    class _FakeOrchestrator:
        def __init__(self, *, output_root):
            self.output_root = output_root

        def execute_jobs(self, jobs, **kwargs):
            return SimpleNamespace(
                reports=[],
                total_jobs=1,
                skipped_runs=0,
                scheduled_runs=1,
                failed_reports=[],
                interrupted=True,
            )

    monkeypatch.setattr("mobilesfrdth.simulator.engine.GridRunOrchestrator", _FakeOrchestrator)

    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "out"),
            "--profile",
            "smoke",
        ]
    )
    assert cli.cmd_run(args) == 130
