from pathlib import Path

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
