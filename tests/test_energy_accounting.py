import pytest
from loraflexsim.launcher.simulator import Simulator


def test_tx_energy_accounted_once():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        fixed_sf=7,
        fixed_tx_power=14.0,
        seed=0,
    )
    sim.run()
    node = sim.nodes[0]
    current = node.profile.get_tx_current(node.tx_power)
    airtime = node.channel.airtime(node.sf, payload_size=sim.payload_size_bytes)
    expected_tx = current * node.profile.voltage_v * airtime
    assert node.energy_tx == pytest.approx(expected_tx)
    expected_ramp_tx = current * node.profile.voltage_v * (
        node.profile.ramp_up_s + node.profile.ramp_down_s
    )
    rx_current = (
        node.profile.listen_current_a
        if node.profile.listen_current_a > 0.0
        else node.profile.rx_current_a
    )
    window_duration = node.profile.rx_window_duration
    expected_rx_window = (
        rx_current * node.profile.voltage_v * window_duration * 2
        if window_duration > 0.0
        else 0.0
    )
    expected_ramp_rx = rx_current * node.profile.voltage_v * (
        node.profile.ramp_up_s + node.profile.ramp_down_s
    )
    assert node.energy_ramp == pytest.approx(
        expected_ramp_tx + 2 * expected_ramp_rx
    )
    assert node.energy_rx == pytest.approx(expected_rx_window)
    expected_startup = (
        node.profile.startup_current_a
        * node.profile.voltage_v
        * node.profile.startup_time_s
        * 2
    )
    expected_preamble = (
        node.profile.preamble_current_a
        * node.profile.voltage_v
        * node.profile.preamble_time_s
    )
    assert node.energy_startup == pytest.approx(expected_startup)
    assert node.energy_preamble == pytest.approx(expected_preamble)
    metrics = sim.get_metrics()
    breakdown = metrics["energy_breakdown_by_node"][node.id]
    assert breakdown.get("rx", 0.0) == pytest.approx(node.energy_rx)
    assert metrics["energy_nodes_J"] == pytest.approx(node.energy_consumed)


def test_downlink_energy_uses_gateway_power():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=0,
        mobility=False,
        fixed_sf=7,
        seed=0,
        node_class="C",
    )
    node = sim.nodes[0]
    gw = sim.gateways[0]
    gw.downlink_power_dBm = 20.0
    sim.network_server.send_downlink(node, b"x")
    for _ in range(10):
        if not sim.step():
            break
        if gw.energy_tx > 0:
            break
    current = gw.profile.get_tx_current(gw.downlink_power_dBm)
    airtime = node.channel.airtime(node.sf, len(b"x"))
    expected_tx = current * gw.profile.voltage_v * airtime
    expected_ramp = current * gw.profile.voltage_v * (
        gw.profile.ramp_up_s + gw.profile.ramp_down_s
    )
    assert gw.energy_tx == pytest.approx(expected_tx)
    assert gw.energy_ramp == pytest.approx(expected_ramp)
