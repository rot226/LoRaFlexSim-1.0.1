"""Scénario IWCMC SNIR statique S3."""

from __future__ import annotations

from pathlib import Path

from snir_static_common import ScenarioConfig, run_scenario

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "S3.csv"

CONFIG = ScenarioConfig(
    figure_id="S3",
    radius_km=2.5,
    node_counts=(400, 1200, 2400),
    packet_interval_s=300.0,
    packets_per_node=10,
    seeds=(7, 8, 9),
    pdr_targets=(0.9, 0.8, 0.7),
    output_path=OUTPUT_PATH,
)


if __name__ == "__main__":
    run_scenario(CONFIG)
