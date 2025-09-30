"""Integration test ensuring long range preset matches FLoRa traces."""

from __future__ import annotations

from pathlib import Path

import pytest

try:  # pragma: no cover - optional dependency
    pytest.importorskip("pandas")
except Exception:  # pragma: no cover - skip gracefully when pandas is unusable
    pytest.skip("pandas import failed", allow_module_level=True)

from loraflexsim.launcher.config_loader import load_config
from loraflexsim.scenarios import build_long_range_simulator
from loraflexsim.validation import (
    ScenarioTolerance,
    compare_to_reference,
    load_flora_reference,
    run_validation,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
FLORA_CONFIG = ROOT_DIR / "flora-master" / "simulations" / "examples" / "long_range_flora.ini"
FLORA_REFERENCE = DATA_DIR / "long_range_flora.sca"


def _distances(nodes: list[dict], gateway: dict) -> list[float]:
    gx = gateway["x"]
    gy = gateway["y"]
    return [((node["x"] - gx) ** 2 + (node["y"] - gy) ** 2) ** 0.5 for node in nodes]


@pytest.mark.slow
def test_long_range_flora_parity_matches_reference() -> None:
    """build_long_range_simulator should reproduce the FLoRa long range trace."""

    nodes_cfg, gateways_cfg, _, _ = load_config(FLORA_CONFIG)
    assert gateways_cfg, "Reference scenario must define a gateway"
    assert len(gateways_cfg) == 1, "Reference scenario assumes a single gateway"

    simulator = build_long_range_simulator("flora", seed=3)
    assert simulator.flora_mode is True
    assert len(nodes_cfg) == len(simulator.nodes)

    gateway = simulator.gateways[0]
    cfg_gateway = gateways_cfg[0]
    assert gateway.x == pytest.approx(cfg_gateway["x"])
    assert gateway.y == pytest.approx(cfg_gateway["y"])

    expected_distances = _distances(nodes_cfg, cfg_gateway)
    observed_distances = [
        ((node.x - gateway.x) ** 2 + (node.y - gateway.y) ** 2) ** 0.5
        for node in simulator.nodes
    ]
    assert observed_distances == pytest.approx(expected_distances)

    expected_sf = [node["sf"] for node in nodes_cfg]
    assert [node.sf for node in simulator.nodes] == expected_sf

    expected_power = [node["tx_power"] for node in nodes_cfg]
    assert [node.tx_power for node in simulator.nodes] == expected_power

    metrics = run_validation(simulator)
    reference = load_flora_reference(FLORA_REFERENCE)
    tolerance = ScenarioTolerance(pdr=0.01, collisions=0, snr=0.2)
    deltas = compare_to_reference(metrics, reference, tolerance)

    assert deltas["PDR"] <= tolerance.pdr
    assert deltas["collisions"] <= tolerance.collisions
    assert deltas["snr"] <= tolerance.snr
    assert metrics["PDR"] == pytest.approx(reference["PDR"], abs=tolerance.pdr)
