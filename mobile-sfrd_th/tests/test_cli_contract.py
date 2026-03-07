import json
from pathlib import Path

import mobilesfrdth.cli as cli
from mobilesfrdth.cli import main
from mobilesfrdth.plotting.plots import validate_aggregates_inputs


def test_plots_returns_non_zero_if_required_csv_missing(tmp_path: Path):
    aggregates = tmp_path / "aggregates"
    aggregates.mkdir(parents=True)

    code = main(["plots", "--aggregates-dir", str(aggregates), "--out", str(tmp_path / "plots")])

    assert code != 0


def test_validate_aggregates_inputs_reports_missing_columns(tmp_path: Path):
    aggregates = tmp_path / "aggregates"
    aggregates.mkdir(parents=True)

    (aggregates / "metric_by_factor.csv").write_text("N,algo\n50,ucb\n", encoding="utf-8")
    (aggregates / "distribution_sf.csv").write_text("algo,sf,ratio\nucb,7,0.5\n", encoding="utf-8")
    (aggregates / "convergence_tc.csv").write_text("algo,speed,Tc_s\nucb,1.0,10\n", encoding="utf-8")
    (aggregates / "sinr_cdf.csv").write_text("algo,quantile,sinr_db\nucb,0.5,2.0\n", encoding="utf-8")
    (aggregates / "fairness_airtime_switching.csv").write_text(
        "N,algo,jain_fairness,airtime_total_s,switch_count\n50,ucb,0.9,12,1\n",
        encoding="utf-8",
    )

    errors = validate_aggregates_inputs(aggregates)

    assert any("metric_by_factor.csv" in err for err in errors)
    assert any("mode" in err for err in errors)


def test_aggregate_returns_non_zero_when_no_run_found(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()

    code = main(["aggregate", "--results", str(empty), "--out", str(tmp_path / "out")])

    assert code != 0


def test_verbose_and_quiet_are_mutually_exclusive(tmp_path: Path):
    aggregates = tmp_path / "aggregates"
    aggregates.mkdir()

    code = main(["--verbose", "--quiet", "plots", "--aggregates-dir", str(aggregates), "--out", str(tmp_path / "plots")])

    assert code != 0


def test_keyboard_interrupt_in_aggregate_writes_partial_and_returns_130(tmp_path: Path, monkeypatch):
    def _boom(*, inputs, output_root, progress_callback=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "aggregate_runs", _boom)

    out = tmp_path / "agg_out"
    code = main(["aggregate", "--results", str(tmp_path), "--out", str(out)])

    assert code == 130
    partial = out / "aggregate_partial.json"
    assert partial.is_file()
    payload = json.loads(partial.read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"
    assert payload["message"] == "reprendre via --resume"
