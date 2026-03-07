import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mobilesfrdth.simulator.io import SUMMARY_COLUMNS, aggregate_runs


def _write_csv(path: pathlib.Path, headers: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow({header: row.get(header, "") for header in headers})


def _summary_row(run_id: str) -> dict[str, object]:
    return {
        "N": "50",
        "speed": "1",
        "mobility_model": "rwp",
        "mode": "snir_on",
        "algo": "ucb",
        "gateways": "1",
        "sigma": "2",
        "seed": "1",
        "rep": "0",
        "run_id": run_id,
        "duration_s": "100",
        "node_count": "50",
        "tx_count": "10",
        "success_count": "9",
        "generated_packets": "10",
        "delivered_bytes": "450",
        "pdr": "0.9",
        "der": "0.9",
        "throughput_bps": "36",
        "Tc_s": "25",
        "jain_fairness": "0.95",
        "airtime_total_s": "12.5",
        "airtime_mean_per_node_s": "0.25",
        "outage_ratio": "0.1",
        "switch_count": "3",
    }


def test_aggregate_runs_summary_only_reads_only_summary(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run_001"
    _write_csv(run_dir / "summary.csv", SUMMARY_COLUMNS, _summary_row("run_001"))

    files = aggregate_runs(inputs=[tmp_path], output_root=tmp_path / "out", summary_only=True)

    captured = capsys.readouterr()
    assert "Agrégation des runs: 1/1" in captured.out
    assert set(files) == {"metric_by_factor", "convergence_tc", "fairness_airtime_switching"}
    for path in files.values():
        assert path.is_file()


def test_aggregate_runs_skip_flags_control_event_outputs(tmp_path):
    run_dir = tmp_path / "results" / "run_001"
    _write_csv(run_dir / "summary.csv", SUMMARY_COLUMNS, _summary_row("run_001"))
    _write_csv(
        run_dir / "events.csv",
        ["event_type", "N", "speed", "mobility_model", "mode", "algo", "gateways", "sigma", "sf", "sinr_db"],
        {
            "event_type": "uplink",
            "N": "50",
            "speed": "1",
            "mobility_model": "rwp",
            "mode": "snir_on",
            "algo": "ucb",
            "gateways": "1",
            "sigma": "2",
            "sf": "9",
            "sinr_db": "6.5",
        },
    )

    files = aggregate_runs(
        inputs=[tmp_path],
        output_root=tmp_path / "out",
        skip_sinr_cdf=True,
        skip_sf_distribution=False,
    )

    assert "distribution_sf" in files
    assert "sinr_cdf" not in files
    assert files["distribution_sf"].is_file()
