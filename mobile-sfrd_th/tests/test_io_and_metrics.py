from pathlib import Path

from mobilesfrdth.simulator.io import aggregate_runs, write_run_outputs
from mobilesfrdth.simulator.metrics import airtime_lora, convergence_tc, der, jain_fairness, outage_ratio, pdr, throughput


def _sample_config(seed: int, rep: int) -> dict:
    return {
        "N": 50,
        "speed": 1.0,
        "mobility_model": "rwp",
        "mode": "adr",
        "algo": "ucb",
        "gateways": 2,
        "sigma": 4.0,
        "seed": seed,
        "rep": rep,
    }


def _sample_events() -> list[dict]:
    return [
        {
            "time_s": 1.0,
            "event_type": "uplink",
            "node_id": 1,
            "sf": 7,
            "sinr_db": 5.0,
            "success": 1,
            "delivered": 1,
            "payload_bytes": 12,
            "airtime_s": 0.05,
            "outage": 0,
            "switch_count": 1,
        },
        {
            "time_s": 3.0,
            "event_type": "uplink",
            "node_id": 2,
            "sf": 8,
            "sinr_db": -2.0,
            "success": 0,
            "delivered": 0,
            "payload_bytes": 12,
            "airtime_s": 0.07,
            "outage": 1,
            "switch_count": 0,
        },
    ]


def test_write_run_outputs_and_aggregate(tmp_path: Path):
    write_run_outputs(
        output_root=tmp_path,
        run_id="run_a",
        run_config=_sample_config(seed=1, rep=0),
        events=_sample_events(),
        duration_s=10.0,
        time_bin_s=5.0,
    )
    write_run_outputs(
        output_root=tmp_path,
        run_id="run_b",
        run_config=_sample_config(seed=2, rep=1),
        events=_sample_events(),
        duration_s=10.0,
        time_bin_s=5.0,
    )

    run_a = tmp_path / "results" / "run_a"
    assert (run_a / "run_config.json").is_file()
    assert (run_a / "events.csv").is_file()
    assert (run_a / "node_timeseries.csv").is_file()
    assert (run_a / "summary.csv").is_file()

    outputs = aggregate_runs(inputs=[tmp_path], output_root=tmp_path)
    assert outputs["metric_by_factor"].is_file()
    assert outputs["distribution_sf"].is_file()
    assert outputs["convergence_tc"].is_file()
    assert outputs["sinr_cdf"].is_file()
    assert outputs["fairness_airtime_switching"].is_file()


def test_metrics_formulas_basics():
    assert pdr(8, 10) == 0.8
    assert der(8, 16) == 0.5
    assert throughput(100, 10.0) == 80.0
    assert outage_ratio(1, 4) == 0.25
    assert round(jain_fairness([1, 1, 1]), 5) == 1.0
    assert airtime_lora(12, sf=7) > 0
    assert convergence_tc([0.2, 0.5, 0.95, 0.97, 0.96], dt_s=1.0, target=1.0, tolerance=0.05, stable_bins=2) == 3.0


def test_aggregate_ignores_incomplete_run_status(tmp_path: Path):
    write_run_outputs(
        output_root=tmp_path,
        run_id="run_ok",
        run_config=_sample_config(seed=1, rep=0),
        events=_sample_events(),
        duration_s=10.0,
        time_bin_s=5.0,
    )
    write_run_outputs(
        output_root=tmp_path,
        run_id="run_partial",
        run_config=_sample_config(seed=2, rep=1),
        events=_sample_events(),
        duration_s=10.0,
        time_bin_s=5.0,
    )

    (tmp_path / "results" / "run_ok" / "run_status.json").write_text(
        "{\"run_id\":\"run_ok\",\"status\":\"completed\"}\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "run_partial" / "run_status.json").write_text(
        "{\"run_id\":\"run_partial\",\"status\":\"interrupted\"}\n",
        encoding="utf-8",
    )

    outputs = aggregate_runs(inputs=[tmp_path], output_root=tmp_path)
    rows = (outputs["convergence_tc"]).read_text(encoding="utf-8").strip().splitlines()

    assert len(rows) == 2
    assert "run_ok" in rows[1]
    assert "run_partial" not in rows[1]
