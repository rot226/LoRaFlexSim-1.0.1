"""Scénario IWCMC SNIR statique S2."""

from __future__ import annotations

from pathlib import Path

from snir_static_common import ScenarioConfig, run_scenario

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "S2.csv"

CONFIG = ScenarioConfig(
    figure_id="S2",
    radius_km=2.5,
    node_counts=(300, 900, 1800),
    packet_interval_s=600.0,
    packets_per_node=10,
    seeds=(4, 5, 6),
    pdr_targets=(0.9, 0.8, 0.7),
    output_path=OUTPUT_PATH,
)


if __name__ == "__main__":
    run_scenario(CONFIG)
