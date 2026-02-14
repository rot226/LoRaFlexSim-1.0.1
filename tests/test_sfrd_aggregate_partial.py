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
