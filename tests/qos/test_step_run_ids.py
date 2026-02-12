from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_step1_scenarios  # noqa: E402  pylint: disable=wrong-import-position
import run_step2_scenarios  # noqa: E402  pylint: disable=wrong-import-position


def test_step1_resume_ignores_completed_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_step1_scenarios,
        "_run_simulation",
        lambda **_: {
            "num_nodes": 100,
            "qos_cluster_pdr": {},
            "sf_distribution": {},
            "collisions": 0,
            "snir_mean": 0.0,
            "snr_mean": 0.0,
        },
    )
    monkeypatch.setattr(
        run_step1_scenarios,
        "_flatten_metrics",
        lambda _: {"collisions": 0, "snir_mean": 0.0, "snr_mean": 0.0},
    )

    duplicate_id = "100-1-8-adr-snir_on"
    rows = run_step1_scenarios._collect_rows(
        nodes=[100],
        charges=[10.0],
        channels=[1],
        algos=["adr"],
        snir_modes=[True],
        replications=1,
        seed=7,
        duration=1.0,
        mixra_solver="auto",
        snir_window=None,
        quiet=True,
        existing_run_ids={duplicate_id},
        resume=True,
        overwrite_run=False,
    )
    assert rows == []

    with pytest.raises(ValueError, match="run_id déjà présent"):
        run_step1_scenarios._collect_rows(
            nodes=[100],
            charges=[10.0],
            channels=[1],
            algos=["adr"],
            snir_modes=[True],
            replications=1,
            seed=7,
            duration=1.0,
            mixra_solver="auto",
            snir_window=None,
            quiet=True,
            existing_run_ids={duplicate_id},
            resume=False,
            overwrite_run=False,
        )


def _write_step2_metrics(path: Path) -> None:
    fieldnames = ["snir_state", "algorithm", "num_nodes", "reward_mean", "pdr"]
    rows = [
        {"snir_state": "snir_on", "algorithm": "ucb1", "num_nodes": "200", "reward_mean": "0.9", "pdr": "0.8"}
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_step2_prevents_duplicate_run_id_unless_resume(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_step2_metrics(input_dir / "metrics.csv")

    run_step2_scenarios.run_normalisation(input_dir, output_dir, quiet=True)

    with pytest.raises(ValueError, match="run_id déjà présent"):
        run_step2_scenarios.run_normalisation(input_dir, output_dir, quiet=True)

    run_step2_scenarios.run_normalisation(input_dir, output_dir, quiet=True, resume=True)

    metrics_path = output_dir / "raw" / "metrics.csv"
    with metrics_path.open("r", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
