from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

run_campaign = importlib.import_module("sfrd.cli.run_campaign")


def _build_args(logs_root: Path, *, force_rerun: bool = False, replications: int = 2) -> Namespace:
    return Namespace(
        network_sizes=[10],
        replications=replications,
        seeds_base=100,
        snir=["OFF"],
        algos=["UCB"],
        warmup_s=0.0,
        ucb_config=Path("sfrd/config/ucb_config.json"),
        logs_root=logs_root,
        skip_aggregate=True,
        resume=True,
        force_rerun=force_rerun,
    )


def test_resume_skips_completed_runs_and_updates_state(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    run1_dir = logs_root / "SNIR_OFF" / "ns_10" / "algo_UCB" / "seed_100"
    run1_dir.mkdir(parents=True, exist_ok=True)
    (run1_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
    (run1_dir / "raw_packets.csv").write_text("h\n", encoding="utf-8")
    (run1_dir / "raw_energy.csv").write_text("h\n", encoding="utf-8")
    (logs_root / "campaign_state.json").write_text(
        json.dumps(
            {
                "runs": {
                    "SNIR_OFF|ns_10|algo_UCB|seed_100": {
                        "snir": "OFF",
                        "network_size": 10,
                        "algo": "UCB",
                        "seed": 100,
                        "status": "done",
                        "paths": {},
                        "duration_s": 1.2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[int] = []

    def _fake_runner(**kwargs):
        seed = int(kwargs["seed"])
        calls.append(seed)
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "raw_packets.csv").write_text("h\n", encoding="utf-8")
        (output_dir / "raw_energy.csv").write_text("h\n", encoding="utf-8")
        return {
            "summary": {"metrics": {"tx_attempted": 1, "rx_delivered": 1, "pdr": 1.0}},
            "summary_path": str(output_dir / "campaign_summary.json"),
        }

    monkeypatch.setattr(run_campaign, "_parse_args", lambda: _build_args(logs_root))
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)

    run_campaign.main()

    assert calls == [101]

    state = json.loads((logs_root / "campaign_state.json").read_text(encoding="utf-8"))
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_100"]["status"] == "done"
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_101"]["status"] == "done"


def test_resume_relaunches_failed_incomplete_and_pending(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    for seed in (100, 101, 102):
        run_dir = logs_root / "SNIR_OFF" / "ns_10" / "algo_UCB" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

    done_dir = logs_root / "SNIR_OFF" / "ns_10" / "algo_UCB" / "seed_103"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
    (done_dir / "raw_packets.csv").write_text("h\n", encoding="utf-8")
    (done_dir / "raw_energy.csv").write_text("h\n", encoding="utf-8")

    (logs_root / "campaign_state.json").write_text(
        json.dumps(
            {
                "runs": {
                    "SNIR_OFF|ns_10|algo_UCB|seed_100": "failed",
                    "SNIR_OFF|ns_10|algo_UCB|seed_101": "incomplete",
                    "SNIR_OFF|ns_10|algo_UCB|seed_102": "pending",
                    "SNIR_OFF|ns_10|algo_UCB|seed_103": "done",
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[int] = []

    def _fake_runner(**kwargs):
        seed = int(kwargs["seed"])
        calls.append(seed)
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "raw_packets.csv").write_text("h\n", encoding="utf-8")
        (output_dir / "raw_energy.csv").write_text("h\n", encoding="utf-8")
        return {
            "summary": {"metrics": {"tx_attempted": 1, "rx_delivered": 1, "pdr": 1.0}},
            "summary_path": str(output_dir / "campaign_summary.json"),
        }

    monkeypatch.setattr(run_campaign, "_parse_args", lambda: _build_args(logs_root, replications=4))
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)

    run_campaign.main()

    assert calls == [100, 101, 102]


def test_force_rerun_ignores_existing_artifacts(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    run1_dir = logs_root / "SNIR_OFF" / "ns_10" / "algo_UCB" / "seed_100"
    run1_dir.mkdir(parents=True, exist_ok=True)
    (run1_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
    (run1_dir / "raw_packets.csv").write_text("h\n", encoding="utf-8")
    (run1_dir / "raw_energy.csv").write_text("h\n", encoding="utf-8")

    (logs_root / "campaign_state.json").write_text(
        json.dumps(
            {
                "runs": {
                    "SNIR_OFF|ns_10|algo_UCB|seed_100": "failed",
                    "SNIR_OFF|ns_10|algo_UCB|seed_101": "pending",
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[int] = []

    def _fake_runner(**kwargs):
        seed = int(kwargs["seed"])
        calls.append(seed)
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "raw_packets.csv").write_text("h\n", encoding="utf-8")
        (output_dir / "raw_energy.csv").write_text("h\n", encoding="utf-8")
        return {
            "summary": {"metrics": {"tx_attempted": 1, "rx_delivered": 1, "pdr": 1.0}},
            "summary_path": str(output_dir / "campaign_summary.json"),
        }

    monkeypatch.setattr(
        run_campaign,
        "_parse_args",
        lambda: _build_args(logs_root, force_rerun=True),
    )
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)

    run_campaign.main()

    assert calls == [100, 101]
    state = json.loads((logs_root / "campaign_state.json").read_text(encoding="utf-8"))
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_100"]["status"] == "done"
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_101"]["status"] == "done"


def test_keyboard_interrupt_marks_run_incomplete(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"

    def _fake_runner(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_campaign, "_parse_args", lambda: _build_args(logs_root, replications=1))
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)

    with pytest.raises(KeyboardInterrupt):
        run_campaign.main()

    state = json.loads((logs_root / "campaign_state.json").read_text(encoding="utf-8"))
    entry = state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_100"]
    assert entry["status"] == "incomplete"
    assert isinstance(entry["duration_s"], float)


def test_run_campaign_aborts_on_raw_vs_summary_pdr_divergence(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"

    def _fake_runner(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "raw_packets.csv").write_text(
            "time,node_id,sf,tx_ok,rx_ok,payload_bytes,run\n"
            "1,0,7,1,0,20,1\n",
            encoding="utf-8",
        )
        (output_dir / "raw_energy.csv").write_text(
            "total_energy_joule,sim_duration_s\n1.0,1.0\n",
            encoding="utf-8",
        )
        return {
            "summary": {"metrics": {"tx_attempted": 1, "rx_delivered": 1, "pdr": 1.0}},
            "summary_path": str(output_dir / "campaign_summary.json"),
        }

    monkeypatch.setattr(run_campaign, "_parse_args", lambda: _build_args(logs_root))
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)

    with pytest.raises(run_campaign.MetricsInconsistentError):
        run_campaign.main()
