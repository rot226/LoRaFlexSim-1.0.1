from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

run_campaign = importlib.import_module("sfrd.cli.run_campaign")


def _find_entry(state: dict, seed: int) -> dict:
    for payload in state.get("runs", {}).values():
        if isinstance(payload, dict) and payload.get("seed") == seed:
            return payload
    raise AssertionError(f"Entrée seed={seed} introuvable dans campaign_state.json")


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
        skip_precheck=True,
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
    assert _find_entry(state, 100)["status"] == "done"
    assert _find_entry(state, 101)["status"] == "done"


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
                        "SNIR_OFF|ns_10|algo_UCB|seed_103": {
                            "snir": "OFF",
                            "network_size": 10,
                            "algo": "UCB",
                            "seed": 103,
                            "status": "done",
                            "paths": {},
                            "duration_s": 1.2,
                        },
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
    assert _find_entry(state, 100)["status"] == "done"
    assert _find_entry(state, 101)["status"] == "done"


def test_keyboard_interrupt_marks_run_incomplete_and_writes_missing_report(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"

    def _fake_runner(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_campaign, "_parse_args", lambda: _build_args(logs_root, replications=1))
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)

    run_campaign.main()

    state = json.loads((logs_root / "campaign_state.json").read_text(encoding="utf-8"))
    entry = _find_entry(state, 100)
    assert entry["status"] == "incomplete"
    assert isinstance(entry["duration_s"], float)

    missing_report = logs_root / "campaign_missing_combinations.csv"
    assert missing_report.exists()
    report_lines = missing_report.read_text(encoding="utf-8").splitlines()
    assert report_lines[0] == "snir,network_size,algorithm,seed,status"
    assert report_lines[1] == "OFF,10,UCB,100,missing"


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


def test_precheck_blocks_campaign_on_invalid_aggregate_csv(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    calls: list[int] = []

    def _fake_runner(**kwargs):
        calls.append(int(kwargs["seed"]))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "raw_packets.csv").write_text(
            "time,node_id,sf,tx_ok,rx_ok,payload_bytes,run\n1,0,7,1,1,20,1\n",
            encoding="utf-8",
        )
        (output_dir / "raw_energy.csv").write_text(
            "total_energy_joule,sim_duration_s\n1.0,1.0\n",
            encoding="utf-8",
        )
        (output_dir / "campaign_summary.json").write_text("{}", encoding="utf-8")
        return {
            "summary": {"metrics": {"tx_attempted": 1, "rx_delivered": 1, "pdr": 1.0}},
            "summary_path": str(output_dir / "campaign_summary.json"),
        }

    def _fake_aggregate(_root: Path, allow_partial: bool):
        assert allow_partial is False
        out = logs_root / "precheck" / "aggregated"
        (out / "SNIR_OFF").mkdir(parents=True, exist_ok=True)
        (out / "SNIR_ON").mkdir(parents=True, exist_ok=True)
        expected = {
            "pdr_results.csv": "network_size,algorithm,snir,pdr\n80,UCB,OFF,1.5\n",
            "throughput_results.csv": "network_size,algorithm,snir,throughput_packets_per_s\n80,UCB,OFF,0.1\n",
            "energy_results.csv": "network_size,algorithm,snir,energy_joule_per_packet\n80,UCB,OFF,1.0\n",
            "sf_distribution.csv": "network_size,algorithm,snir,sf,count\n80,UCB,OFF,7,1\n",
        }
        for rel, body in expected.items():
            (out / "SNIR_OFF" / rel).write_text(body, encoding="utf-8")
            (out / "SNIR_ON" / rel).write_text(body.replace(",OFF,", ",ON,"), encoding="utf-8")
        (out / "learning_curve_ucb.csv").write_text("episode,reward\n1,0.1\n", encoding="utf-8")
        return out

    args = _build_args(logs_root, replications=1)
    args.skip_precheck = False
    monkeypatch.setattr(run_campaign, "_parse_args", lambda: args)
    monkeypatch.setattr(run_campaign, "_load_run_single_campaign", lambda: _fake_runner)
    monkeypatch.setattr(run_campaign, "aggregate_logs", _fake_aggregate)

    with pytest.raises(RuntimeError, match="précheck NO-GO"):
        run_campaign.main()

    assert len(calls) == 4
