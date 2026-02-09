"""Scénario IWCMC SNIR statique S7."""

from __future__ import annotations

from pathlib import Path

from snir_static_common import ScenarioConfig, run_scenario

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "S7.csv"

CONFIG = ScenarioConfig(
    figure_id="S7",
    radius_km=3.5,
    node_counts=(450, 1350, 2700),
    packet_interval_s=300.0,
    packets_per_node=10,
    seeds=(19, 20, 21),
    pdr_targets=(0.9, 0.8, 0.7),
    output_path=OUTPUT_PATH,
)


if __name__ == "__main__":
    run_scenario(CONFIG)
