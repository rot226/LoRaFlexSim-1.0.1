"""Scénario IWCMC SNIR statique S5."""

from __future__ import annotations

from pathlib import Path

from snir_static_common import ScenarioConfig, run_scenario

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "S5.csv"

CONFIG = ScenarioConfig(
    figure_id="S5",
    radius_km=3.5,
    node_counts=(250, 750, 1500),
    packet_interval_s=900.0,
    packets_per_node=10,
    seeds=(13, 14, 15),
    pdr_targets=(0.9, 0.8, 0.7),
    output_path=OUTPUT_PATH,
)


if __name__ == "__main__":
    run_scenario(CONFIG)
