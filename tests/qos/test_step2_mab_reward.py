from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_step2_scenarios  # noqa: E402  pylint: disable=wrong-import-position


def _write_metrics_csv(path: Path) -> None:
    fieldnames = [
        "snir_state",
        "cluster",
        "num_nodes",
        "sf",
        "reward_mean",
        "reward_variance",
        "der",
        "pdr",
        "snir_avg",
        "success_rate",
        "energy_j",
    ]
    rows = [
        {
            "snir_state": "snir_off",
            "cluster": "1",
            "num_nodes": "120",
            "sf": "9",
            "reward_mean": "0.32",
            "reward_variance": "0.01",
            "der": "0.62",
            "pdr": "0.62",
            "snir_avg": "7.5",
            "success_rate": "0.62",
            "energy_j": "0.5",
        },
        {
            "snir_state": "snir_on",
            "cluster": "1",
            "num_nodes": "120",
            "sf": "9",
            "reward_mean": "0.45",
            "reward_variance": "0.02",
            "der": "0.74",
            "pdr": "0.74",
            "snir_avg": "12.1",
            "success_rate": "0.74",
            "energy_j": "0.5",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step2_mab_reward_snir_on_off(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_metrics_csv(input_dir / "ucb1_snir_demo.csv")

    run_step2_scenarios.run_normalisation(input_dir, output_dir, quiet=True)

    metrics_path = output_dir / "raw" / "metrics.csv"
    assert metrics_path.exists(), "Le CSV métriques normalisé doit être produit"

    reward_by_state: dict[str, float] = {}
    with metrics_path.open(newline="", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reward_by_state[row["snir_state"]] = float(row["reward_mean"])

    assert "snir_on" in reward_by_state and "snir_off" in reward_by_state, (
        "Les métriques doivent couvrir SNIR activé et désactivé."
    )
    assert reward_by_state["snir_on"] > reward_by_state["snir_off"], (
        "La récompense MAB doit être supérieure lorsque le SNIR est activé."
    )
