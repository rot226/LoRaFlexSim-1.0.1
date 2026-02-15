from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfrd.parse.aggregate import aggregate_logs


def _write_summary(run_dir: Path, *, snir: str, size: int, algo: str, seed: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": {
            "snir_mode": snir,
            "network_size": size,
            "algorithm": algo,
            "seed": seed,
        },
        "metrics": {
            "pdr": 0.8,
            "throughput_packets_per_s": 1.1,
            "energy_joule_per_packet": 0.2,
            "sf_distribution_counts": {"7": 10},
        },
    }
    (run_dir / "campaign_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_strict_fails_when_expected_runs_are_missing(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    state_payload = {
        "runs": {
            "a": {"snir": "OFF", "network_size": 80, "algo": "UCB", "seed": 1, "status": "done"},
            "b": {"snir": "OFF", "network_size": 80, "algo": "UCB", "seed": 2, "status": "pending"},
        }
    }
    (logs_root / "campaign_state.json").parent.mkdir(parents=True, exist_ok=True)
    (logs_root / "campaign_state.json").write_text(json.dumps(state_payload), encoding="utf-8")
    _write_summary(logs_root / "SNIR_OFF" / "ns_80" / "algo_UCB" / "seed_1", snir="OFF", size=80, algo="UCB", seed=1)

    with pytest.raises(RuntimeError):
        aggregate_logs(logs_root)


def test_aggregate_allow_partial_writes_completeness(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    state_payload = {
        "runs": {
            "a": {"snir": "OFF", "network_size": 80, "algo": "UCB", "seed": 1, "status": "done"},
            "b": {"snir": "OFF", "network_size": 80, "algo": "UCB", "seed": 2, "status": "pending"},
        }
    }
    (logs_root / "campaign_state.json").parent.mkdir(parents=True, exist_ok=True)
    (logs_root / "campaign_state.json").write_text(json.dumps(state_payload), encoding="utf-8")
    _write_summary(logs_root / "SNIR_OFF" / "ns_80" / "algo_UCB" / "seed_1", snir="OFF", size=80, algo="UCB", seed=1)

    output_root = aggregate_logs(logs_root, allow_partial=True)

    completeness = (output_root / "campaign_completeness.csv").read_text(encoding="utf-8")
    assert "OFF,80,UCB,2,1,no,2" in completeness


def test_aggregate_sf_distribution_keeps_all_sf_rows(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    _write_summary(logs_root / "SNIR_OFF" / "ns_80" / "algo_UCB" / "seed_1", snir="OFF", size=80, algo="UCB", seed=1)
    run2 = logs_root / "SNIR_OFF" / "ns_80" / "algo_UCB" / "seed_2"
    run2.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": {"snir_mode": "OFF", "network_size": 80, "algorithm": "UCB", "seed": 2},
        "metrics": {
            "pdr": 0.7,
            "throughput_packets_per_s": 1.0,
            "energy_joule_per_packet": 0.4,
            "sf_distribution_counts": {"7.0": 4, "8": 6, "13": 2},
        },
    }
    (run2 / "campaign_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = aggregate_logs(logs_root, allow_partial=True)

    sf_lines = (output_root / "SNIR_OFF" / "sf_distribution.csv").read_text(encoding="utf-8").splitlines()
    assert sf_lines[0] == "network_size,algorithm,snir,sf,count"
    assert len(sf_lines) == 7
    assert "80,UCB,OFF,7,7.0" in sf_lines
    assert "80,UCB,OFF,8,3.0" in sf_lines
    assert "80,UCB,OFF,9,0.0" in sf_lines
    assert "80,UCB,OFF,10,0.0" in sf_lines
    assert "80,UCB,OFF,11,0.0" in sf_lines
    assert "80,UCB,OFF,12,0.0" in sf_lines


def test_aggregate_writes_missing_combinations_report(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    state_payload = {
        "runs": {
            "a": {"snir": "OFF", "network_size": 80, "algo": "UCB", "seed": 1, "status": "done"},
            "b": {"snir": "OFF", "network_size": 80, "algo": "MixRA-H", "seed": 1, "status": "pending"},
        }
    }
    (logs_root / "campaign_state.json").parent.mkdir(parents=True, exist_ok=True)
    (logs_root / "campaign_state.json").write_text(json.dumps(state_payload), encoding="utf-8")
    _write_summary(logs_root / "SNIR_OFF" / "ns_80" / "algo_UCB" / "seed_1", snir="OFF", size=80, algo="UCB", seed=1)

    output_root = aggregate_logs(logs_root, allow_partial=True)

    missing = (output_root / "campaign_missing_combinations.csv").read_text(encoding="utf-8")
    assert "OFF,80,MixRA-H,1,1" in missing


def test_aggregate_loads_expected_runs_from_missing_report(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    report_path = logs_root / "campaign_missing_combinations.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "snir,network_size,algorithm,seed,status\nOFF,80,MixRA-H,4,missing\n",
        encoding="utf-8",
    )
    _write_summary(logs_root / "SNIR_OFF" / "ns_80" / "algo_UCB" / "seed_1", snir="OFF", size=80, algo="UCB", seed=1)

    output_root = aggregate_logs(logs_root, allow_partial=True)

    completeness = (output_root / "campaign_completeness.csv").read_text(encoding="utf-8")
    assert "OFF,80,MixRA-H,1,0,no,4" in completeness
