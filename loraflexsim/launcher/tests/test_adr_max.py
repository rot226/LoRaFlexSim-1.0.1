from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher import adr_max, ADR_MODULES
import loraflexsim.launcher.server as server


def test_adr_max_apply_configures_simulator():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=False,
        adr_server=False,
        seed=1,
    )
    adr_max.apply(sim)
    node = sim.nodes[0]
    assert "ADR-Max" in ADR_MODULES
    assert sim.adr_node is True
    assert sim.adr_server is True
    assert sim.network_server.adr_enabled is True
    assert sim.network_server.adr_method == "adr-max"
    expected_thr = Channel.flora_detection_threshold(
        node.sf, node.channel.bandwidth
    ) + node.channel.sensitivity_margin_dB
    assert node.channel.detection_threshold_dBm == expected_thr
    assert node.tx_power == 14.0
    assert node.adr_ack_limit == 64
    assert node.adr_ack_delay == 32


def test_adr_max_progressively_reduces_rate():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=False,
        adr_server=False,
        seed=1,
    )
    adr_max.apply(sim)
    node = sim.nodes[0]
    ns = sim.network_server
    noise = ns.channel.noise_floor_dBm()

    initial_sf = node.sf
    initial_power = node.tx_power

    for cycle in range(3):
        rssi = noise + server.REQUIRED_SNR[node.sf] + server.MARGIN_DB + 6.0
        for i in range(20):
            ns.receive(cycle * 20 + i + 1, node.id, sim.gateways[0].id, rssi)
        if cycle == 0:
            assert node.sf == initial_sf - 2
            assert node.tx_power == initial_power
        elif cycle == 1:
            assert node.sf == initial_sf - 4
            assert node.tx_power == initial_power
        else:
            assert node.sf == 7
            assert node.tx_power == initial_power - 2.0
        assert node.snr_history == []


def test_adr_max_margin_uses_max_snr_history():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=False,
        adr_server=False,
        seed=1,
    )
    adr_max.apply(sim)
    node = sim.nodes[0]
    ns = sim.network_server
    noise = ns.channel.noise_floor_dBm()

    low = noise + server.REQUIRED_SNR[node.sf] + server.MARGIN_DB - 2.0
    high = noise + server.REQUIRED_SNR[node.sf] + server.MARGIN_DB + 6.0

    ns.receive(1, node.id, sim.gateways[0].id, high)
    for i in range(2, 21):
        ns.receive(i, node.id, sim.gateways[0].id, low)

    assert node.sf == node.initial_sf - 2
    assert node.tx_power == node.initial_tx_power
    assert node.snr_history == []
