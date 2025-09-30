"""Regression test for the FLoRa log-normal shadowing model."""

import math

import pytest

from loraflexsim.launcher.channel import Channel


pytestmark = pytest.mark.propagation_campaign


@pytest.mark.parametrize(
    "distance",
    [
        40.0,
        80.0,
        200.0,
        1000.0,
        5000.0,
    ],
)
def test_flora_path_loss_matches_flora_reference(distance: float) -> None:
    """Ensure Channel reproduces FLoRa's LoRaLogNormalShadowing curve."""

    channel = Channel(environment="flora", phy_model="flora")
    loss = channel.path_loss(distance)

    gamma = 2.08
    d0 = 40.0
    pl_d0 = 127.41
    expected = pl_d0 + 10 * gamma * math.log10(distance / d0)

    assert loss == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_rural_long_range_rssi_alignment() -> None:
    """Le preset longue portée vise les seuils FLoRa entre 10 et 15 km."""

    channel = Channel(environment="rural_long_range")
    channel.shadowing_std = 0.0
    target_ranges = {
        5000.0: (-121.0, -118.0),
        10000.0: (-128.0, -122.0),
        15000.0: (-131.0, -123.0),
    }
    for distance, (low, high) in target_ranges.items():
        rssi, _ = channel.compute_rssi(14.0, distance, sf=12)
        expected = 14.0 - channel.path_loss(distance)
        assert rssi == pytest.approx(expected, abs=1e-6)
        assert low <= rssi <= high
