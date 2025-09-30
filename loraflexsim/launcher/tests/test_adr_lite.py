from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.lorawan import TX_POWER_INDEX_TO_DBM
from loraflexsim.launcher import adr_lite, ADR_MODULES
import loraflexsim.launcher.server as server


def test_adr_lite_apply_configures_simulator():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        mobility=False,
        adr_node=False,
        adr_server=False,
        duty_cycle=None,
        seed=1,
    )
    adr_lite.apply(sim)
    node = sim.nodes[0]
    assert "ADR-Lite" in ADR_MODULES
    assert sim.adr_node is True
    assert sim.adr_server is True
    assert sim.network_server.adr_enabled is True
    assert sim.network_server.adr_method == "adr-lite"
    assert node.sf == 12
    expected_thr = Channel.flora_detection_threshold(12, node.channel.bandwidth) + node.channel.sensitivity_margin_dB
    assert node.channel.detection_threshold_dBm == expected_thr
    assert node.tx_power == TX_POWER_INDEX_TO_DBM[0]
    assert node.adr_ack_limit == 64
    assert node.adr_ack_delay == 32


def test_adr_lite_server_quick_reaction():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        mobility=False,
        adr_node=False,
        adr_server=False,
        duty_cycle=None,
        seed=1,
    )
    adr_lite.apply(sim)
    node = sim.nodes[0]
    ns = sim.network_server
    expected = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
    assert server.REQUIRED_SNR == expected
    # SNR above the requirement for SF7 to force an immediate downgrade
    noise = ns.channel.noise_floor_dBm()
    rssi = noise + server.REQUIRED_SNR[7] + server.MARGIN_DB + 5.0
    ns.receive(1, node.id, sim.gateways[0].id, rssi)
    assert node.sf == 7
