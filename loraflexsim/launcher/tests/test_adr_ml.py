from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher import adr_ml, ADR_MODULES


def test_adr_ml_apply_configures_simulator():
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
    adr_ml.apply(sim)
    node = sim.nodes[0]
    assert "ADR_ML" in ADR_MODULES
    assert sim.adr_node is True
    assert sim.adr_server is True
    assert sim.network_server.adr_enabled is True
    assert node.sf == 9
    expected_thr = (
        Channel.flora_detection_threshold(9, node.channel.bandwidth)
        + node.channel.sensitivity_margin_dB
    )
    assert node.channel.detection_threshold_dBm == expected_thr
    assert node.tx_power == 14.0
    assert node.adr_ack_limit == 16
    assert node.adr_ack_delay == 12

