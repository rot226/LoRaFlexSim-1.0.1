import pytest

from loraflexsim.launcher.adr_standard_1 import apply as apply_adr
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.server import ADR_WINDOW_SIZE
from loraflexsim.launcher.simulator import Simulator
from loraflexsim.tests.reference_traces import _flora_adr_decision


@pytest.mark.parametrize("node_class", ["B", "C"])
def test_adr_class_bc_reduces_sf_and_power_on_good_link(node_class: str) -> None:
    """Les classes B/C doivent suivre la décision ADR FLoRa sur lien favorable."""

    channel = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        node_class=node_class,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=ADR_WINDOW_SIZE,
        mobility=False,
        adr_server=True,
        adr_method="avg",
        channels=[channel],
        seed=1,
        tick_ns=1_000_000,
    )
    apply_adr(sim)

    node = sim.nodes[0]
    gateway = sim.gateways[0]
    node.x = node.y = 0.0
    gateway.x = gateway.y = 0.0

    initial_sf = node.initial_sf
    initial_power = node.initial_tx_power

    sim.run()

    assert len(node.snr_history) == ADR_WINDOW_SIZE
    snr_values = tuple(snr for _, snr in node.snr_history)
    expected = _flora_adr_decision(
        snr_values,
        initial_sf,
        initial_power,
        method="avg",
    )
    assert expected is not None
    expected_sf, expected_power, *_ = expected

    assert node.sf == expected_sf
    assert node.tx_power == pytest.approx(expected_power, abs=1e-6)
    # Vérifie que l'adaptation a réduit au moins un des paramètres.
    assert expected_sf < initial_sf or expected_power < initial_power


@pytest.mark.parametrize("node_class", ["B", "C"])
def test_adr_class_bc_increases_sf_on_poor_link(node_class: str) -> None:
    """Lien défavorable : l'ADR doit augmenter le SF comme dans FLoRa."""

    channel = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        node_class=node_class,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=ADR_WINDOW_SIZE + 1,
        mobility=False,
        adr_server=True,
        adr_method="avg",
        channels=[channel],
        seed=1,
        tick_ns=1_000_000,
        fixed_sf=9,
    )
    apply_adr(sim)

    node = sim.nodes[0]
    gateway = sim.gateways[0]
    node.x = node.y = 0.0
    gateway.x = 100_000.0
    gateway.y = 0.0

    initial_sf = node.initial_sf
    initial_power = node.initial_tx_power

    sim.run()

    assert len(node.snr_history) == ADR_WINDOW_SIZE
    snr_values = tuple(snr for _, snr in node.snr_history)
    expected = _flora_adr_decision(
        snr_values,
        initial_sf,
        initial_power,
        method="avg",
    )
    assert expected is not None
    expected_sf, expected_power, *_ = expected

    assert node.sf == expected_sf
    assert node.tx_power == pytest.approx(expected_power, abs=1e-6)
    # Le lien est difficile : l'ADR doit augmenter le SF ou la puissance.
    assert expected_sf > initial_sf or expected_power > initial_power
