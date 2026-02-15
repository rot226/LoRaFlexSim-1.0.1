from __future__ import annotations

import json

from sfrd.parse.parse_run import parse_run


def test_parse_run_filters_warmup_and_computes_metrics_csv(tmp_path):
    raw = tmp_path / "run.csv"
    raw.write_text(
        "time,event_type,result,final_sf\n"
        "5,tx,success,7\n"
        "12,tx,collision,8\n"
        "20,tx,success,8\n",
        encoding="utf-8",
    )

    metrics = parse_run(
        raw,
        warmup_s=10.0,
        sim_duration_s=40.0,
        total_energy_joule=10.0,
    )

    assert metrics["tx_count"] == 2
    assert metrics["success_count"] == 1
    assert metrics["effective_duration_s"] == 30.0
    assert metrics["pdr"] == 0.5
    assert metrics["throughput_packets_per_s"] == 1.0 / 30.0
    assert metrics["energy_joule_per_packet"] == 10.0
    assert metrics["sf_distribution"] == {7: 0, 8: 2, 9: 0, 10: 0, 11: 0, 12: 0}


def test_parse_run_no_extrapolation_when_tx_or_duration_or_success_missing_jsonl(tmp_path):
    raw = tmp_path / "run.jsonl"
    events = [
        {"time": 1, "event_type": "rx", "result": "success", "final_sf": 7},
        {"time": 2, "event_type": "rx", "result": "collision", "final_sf": 8},
    ]
    raw.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")

    metrics = parse_run(raw, warmup_s=0.0, sim_duration_s=None, total_energy_joule=5.0)

    assert metrics["tx_count"] == 0
    assert metrics["success_count"] == 0
    assert metrics["effective_duration_s"] == 1.0
    assert metrics["pdr"] is None
    assert metrics["throughput_packets_per_s"] == 0.0
    assert metrics["energy_joule_per_packet"] is None
    assert metrics["sf_distribution"] == {7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}


def test_parse_run_supports_raw_packets_rx_ok_column(tmp_path):
    raw = tmp_path / "raw_packets.csv"
    raw.write_text(
        "time,node_id,sf,tx_ok,rx_ok,payload_bytes,run\n"
        "10,0,7,1,1,20,1\n"
        "20,1,8,1,0,20,1\n",
        encoding="utf-8",
    )

    metrics = parse_run(raw, warmup_s=0.0, sim_duration_s=30.0, total_energy_joule=4.0)

    assert metrics["tx_count"] == 2
    assert metrics["success_count"] == 1
    assert metrics["pdr"] == 0.5


def test_parse_run_accepts_mixed_sf_types_from_raw_packets_csv(tmp_path):
    raw = tmp_path / "raw_packets.csv"
    raw.write_text(
        "time,event_type,result,sf\n"
        "1,tx,success,7\n"
        "2,tx,collision,7.0\n"
        '3,tx,success,"8"\n'
        "4,tx,success,8.4\n"
        "5,tx,success,13\n",
        encoding="utf-8",
    )

    metrics = parse_run(raw, warmup_s=0.0, sim_duration_s=10.0, total_energy_joule=6.0)

    assert metrics["tx_count"] == 5
    assert metrics["sf_distribution"] == {7: 2, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0}
