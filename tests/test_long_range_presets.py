"""Validation longue portée des presets FLoRa."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from loraflexsim.launcher import Channel, Simulator


pytestmark = pytest.mark.propagation_campaign


def _iter_presets() -> Iterator[str]:
    yield from ("flora", "flora_hata", "flora_oulu")


def _make_channel(environment: str) -> Channel:
    loss_model = "lognorm"
    if environment == "flora_hata":
        loss_model = "hata"
    elif environment == "flora_oulu":
        loss_model = "oulu"
    channel = Channel(environment=environment, flora_loss_model=loss_model)
    # Supprimer le shadowing pour stabiliser le scénario de validation.
    channel.shadowing_std = 0.0
    return channel


@pytest.fixture(scope="module")
def long_range_validation() -> tuple[dict[str, float], float]:
    """Exécute un scénario > 5 km et renvoie les PDR mesurés."""

    metrics: dict[str, float] = {}
    max_distance = 0.0
    for env in _iter_presets():
        channel = _make_channel(env)
        simulator = Simulator(
            num_nodes=30,
            num_gateways=1,
            area_size=8000.0,
            packets_to_send=2,
            transmission_mode="Periodic",
            packet_interval=600.0,
            fixed_sf=12,
            fixed_tx_power=14.0,
            mobility=False,
            seed=123,
            flora_mode=True,
            channels=[channel],
        )
        simulator.run()
        metrics[env] = simulator.get_metrics()["PDR"]
        gateway = simulator.gateways[0]
        farthest = max(
            math.hypot(node.x - gateway.x, node.y - gateway.y)
            for node in simulator.nodes
        )
        max_distance = max(max_distance, farthest)
    return metrics, max_distance


def test_long_range_presets_pdr(long_range_validation: tuple[dict[str, float], float]) -> None:
    """Les presets FLoRa restent utilisables au-delà de 5 km."""

    metrics, max_distance = long_range_validation
    assert max_distance > 5000.0
    assert set(metrics) == {"flora", "flora_hata", "flora_oulu"}
    for env, pdr in metrics.items():
        assert 0.0 <= pdr <= 1.0, env
    assert metrics["flora_oulu"] >= metrics["flora_hata"]
    assert metrics["flora_oulu"] > metrics["flora"]
    assert metrics["flora_hata"] >= metrics["flora"]
