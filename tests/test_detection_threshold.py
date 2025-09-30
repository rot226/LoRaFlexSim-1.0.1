from loraflexsim.launcher import adr_standard_1
from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.channel import Channel


def test_apply_sets_flora_detection_threshold():
    sim = Simulator(num_nodes=1, packets_to_send=0)
    adr_standard_1.apply(sim, degrade_channel=True, profile="flora")
    node = sim.nodes[0]
    expected = Channel.FLORA_SENSITIVITY[node.sf][int(node.channel.bandwidth)]
    assert node.channel.detection_threshold_dBm == expected


def test_apply_sets_threshold_without_degradation():
    sim = Simulator(num_nodes=1, packets_to_send=0)
    adr_standard_1.apply(sim, degrade_channel=False)
    node = sim.nodes[0]
    expected = Channel.FLORA_SENSITIVITY[node.sf][int(node.channel.bandwidth)]
    assert node.channel.detection_threshold_dBm == expected


def test_schedule_event_updates_threshold():
    ch = Channel()
    sim = Simulator(num_nodes=1, packets_to_send=0, channels=[ch])
    node = sim.nodes[0]
    node.sf = 9
    sim.schedule_event(node, 0)
    expected = Channel.FLORA_SENSITIVITY[9][int(node.channel.bandwidth)]
    assert node.channel.detection_threshold_dBm == expected


def test_sensitivity_margin_offset():
    ch = Channel(sensitivity_margin_dB=5.0)
    sim = Simulator(num_nodes=1, packets_to_send=0, channels=[ch])
    node = sim.nodes[0]
    node.sf = 10
    sim.schedule_event(node, 0)
    expected = Channel.FLORA_SENSITIVITY[10][int(node.channel.bandwidth)] + 5.0
    assert node.channel.detection_threshold_dBm == expected
