from __future__ import annotations

import json
import importlib
from argparse import Namespace
from pathlib import Path

run_campaign = importlib.import_module("sfrd.cli.run_campaign")


def _build_args(logs_root: Path, *, force_rerun: bool = False) -> Namespace:
    return Namespace(
        network_sizes=[10],
        replications=2,
        seeds_base=100,
        snir=["OFF"],
        algos=["UCB"],
        warmup_s=0.0,
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
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_100"] == "done"
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_101"] == "done"


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
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_100"] == "done"
    assert state["runs"]["SNIR_OFF|ns_10|algo_UCB|seed_101"] == "done"
