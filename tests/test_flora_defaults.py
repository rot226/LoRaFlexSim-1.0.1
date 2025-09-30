"""Tests ensuring FLoRa configurations use the non-orthogonal capture matrix by default."""

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.gateway import FLORA_NON_ORTH_DELTA
from loraflexsim.launcher.simulator import Simulator


def test_flora_simulator_uses_non_orthogonal_capture():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        flora_mode=True,
        phy_model="omnet_full",
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        warm_up_intervals=0,
        log_mean_after=None,
        seed=1234,
    )

    # The main channel and all nodes should immediately use the FLoRa matrix
    assert sim.channel.orthogonal_sf is False
    assert sim.channel.non_orth_delta == FLORA_NON_ORTH_DELTA
    expected_floor = Channel.flora_energy_threshold(sim.channel.bandwidth)
    assert sim.channel.energy_detection_dBm == expected_floor

    node = sim.nodes[0]
    assert node.channel.orthogonal_sf is False
    assert node.channel.non_orth_delta == FLORA_NON_ORTH_DELTA
    assert node.channel.energy_detection_dBm == expected_floor

    gw = sim.gateways[0]
    assert gw.energy_detection_dBm == expected_floor
    assert sim.network_server.energy_detection_dBm == expected_floor


def test_flora_energy_detection_filters_frames():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        flora_mode=True,
        phy_model="omnet_full",
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        warm_up_intervals=0,
        log_mean_after=None,
        seed=42,
    )

    node = sim.nodes[0]
    gateway = sim.gateways[0]

    floor = Channel.flora_energy_threshold(node.channel.bandwidth)
    assert node.channel.energy_detection_dBm == floor
    assert gateway.energy_detection_dBm == floor

    add_called = False
    start_called = False

    def fake_add(*args, **kwargs):
        nonlocal add_called
        add_called = True

    def fake_start(*args, **kwargs):
        nonlocal start_called
        start_called = True

    def fake_average_power(
        gateway_id, frequency, current_time, end_time, *, base_noise_mW=0.0
    ) -> float:
        return base_noise_mW

    def fake_compute_rssi(tx_power, distance, sf, **kwargs):
        node.channel.last_noise_dBm = -100.0
        return -95.0, -5.0

    sim._tx_manager.add = fake_add
    sim._tx_manager.average_power = fake_average_power
    gateway.start_reception = fake_start
    node.channel.compute_rssi = fake_compute_rssi

    sim.run(max_steps=10)
    sim.stop()

    assert not add_called
    assert not start_called
    assert node.last_rssi is None
