"""Integration checks for the large area long range scenario."""

from __future__ import annotations

import math

import pytest

from loraflexsim.launcher import Channel, Simulator
from loraflexsim.scenarios import (
    LONG_RANGE_AREA_SIZE,
    LONG_RANGE_BANDWIDTHS,
    LONG_RANGE_DISTANCES,
    LONG_RANGE_RECOMMENDATIONS,
    LONG_RANGE_SPREADING_FACTORS,
    LongRangeParameters,
    build_simulator_from_suggestion,
    build_long_range_simulator,
    suggest_parameters,
)


def _expected_distances(params: LongRangeParameters) -> list[float]:
    return list(params.distances or tuple(LONG_RANGE_DISTANCES))


def _expected_spreading_factors(params: LongRangeParameters) -> list[int]:
    return list(params.spreading_factors or tuple(LONG_RANGE_SPREADING_FACTORS))


def _expected_area_size(params: LongRangeParameters) -> float:
    return params.area_size_m or LONG_RANGE_AREA_SIZE


@pytest.mark.parametrize("preset", list(LONG_RANGE_RECOMMENDATIONS))
def test_long_range_large_area_meets_pdr_and_sensitivity(preset: str) -> None:
    """Ensure the long range scenario preserves coverage at 10–12 km."""

    params = LONG_RANGE_RECOMMENDATIONS[preset]
    distances_cfg = _expected_distances(params)
    spreading_cfg = _expected_spreading_factors(params)
    area_size = _expected_area_size(params)
    simulator = build_long_range_simulator(preset, seed=3, packets_per_node=params.packets_per_node)
    simulator.run()
    metrics = simulator.get_metrics()

    # Area check (square side length squared) expressed in square metres.
    assert pytest.approx(simulator.area_size, rel=1e-6) == area_size
    assert simulator.area_size / 2.0 >= max(distances_cfg)

    gateway = simulator.gateways[0]
    distances = [
        math.hypot(node.x - gateway.x, node.y - gateway.y) for node in simulator.nodes
    ]
    assert len(distances) == len(distances_cfg)
    for expected, observed in zip(distances_cfg, distances):
        assert observed == pytest.approx(expected)
    assert sorted(node.sf for node in simulator.nodes) == sorted(spreading_cfg)

    sf12_pdr = metrics["pdr_by_sf"][12]
    assert sf12_pdr >= 0.7

    # Gather all successful receptions to verify the sensitivity margins.
    successful_sf12 = [
        ev
        for ev in simulator.events_log
        if ev.get("result") == "Success" and ev["sf"] == 12 and ev["rssi_dBm"] is not None
    ]
    assert successful_sf12, "No SF12 receptions recorded"

    maxima: dict[int, dict[str, float]] = {}
    for event in successful_sf12:
        node = next(n for n in simulator.nodes if n.id == event["node_id"])
        bandwidth = int(node.channel.bandwidth)
        stats = maxima.setdefault(bandwidth, {"rssi": -float("inf"), "snr": -float("inf")})
        stats["rssi"] = max(stats["rssi"], event["rssi_dBm"])
        stats["snr"] = max(stats["snr"], event["snr_dB"])

    expected_bandwidths = {int(bw) for bw in LONG_RANGE_BANDWIDTHS}
    assert set(maxima) == expected_bandwidths

    for bandwidth, stats in maxima.items():
        sensitivity = Channel.FLORA_SENSITIVITY[12][bandwidth]
        assert stats["rssi"] >= sensitivity - 0.1
        assert stats["snr"] >= Simulator.REQUIRED_SNR[12]


def test_very_long_range_extends_distance() -> None:
    """The dedicated preset must include nodes beyond 13 km."""

    params = LONG_RANGE_RECOMMENDATIONS["very_long_range"]
    simulator = build_long_range_simulator("very_long_range", seed=3, packets_per_node=params.packets_per_node)
    simulator.run()

    gateway = simulator.gateways[0]
    distances = [
        math.hypot(node.x - gateway.x, node.y - gateway.y) for node in simulator.nodes
    ]
    assert max(distances) >= 15_000.0
    assert len([d for d in distances if d >= 13_000.0]) >= 2


def test_auto_suggestion_preserves_sf12_reliability() -> None:
    """Auto suggestions must stay within known link budgets and keep SF12 reliable."""

    suggestion = suggest_parameters(area_km2=10.0)
    params = suggestion.parameters

    rural = LONG_RANGE_RECOMMENDATIONS["rural_long_range"]
    very = LONG_RANGE_RECOMMENDATIONS["very_long_range"]

    assert rural.tx_power_dBm <= params.tx_power_dBm <= very.tx_power_dBm
    assert rural.tx_antenna_gain_dB <= params.tx_antenna_gain_dB <= very.tx_antenna_gain_dB
    assert rural.rx_antenna_gain_dB <= params.rx_antenna_gain_dB <= very.rx_antenna_gain_dB
    assert suggestion.area_km2 >= 10.0

    simulator = build_simulator_from_suggestion(suggestion, seed=3)
    simulator.run()
    metrics = simulator.get_metrics()
    assert metrics["pdr_by_sf"][12] >= 0.7
