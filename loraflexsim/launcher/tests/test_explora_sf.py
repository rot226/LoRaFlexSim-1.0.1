import random
from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher import explora_sf
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.channel import Channel


def test_explora_sf_assigns_uniform_groups():
    random.seed(0)
    sim = Simulator(
        num_nodes=12,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=False,
        adr_server=False,
        seed=42,
    )
    explora_sf.apply(sim)
    gw = sim.gateways[0]
    rssis = [-40, -41, -55, -56, -70, -71, -85, -86, -100, -101, -115, -116]
    for i, (node, rssi) in enumerate(zip(sim.nodes, rssis)):
        frame = node.prepare_uplink(b"ping")
        sim.network_server.receive(i, node.id, gw.id, rssi, frame)
    for node in sim.nodes:
        dl = gw.pop_downlink(node.id)
        if dl is not None:
            node.handle_downlink(dl[0])
    sf_counts = {sf: 0 for sf in range(7, 13)}
    for node in sim.nodes:
        sf_counts[node.sf] += 1
    assert all(count == 2 for count in sf_counts.values())
    for i in range(0, len(sim.nodes), 2):
        assert sim.nodes[i].sf == sim.nodes[i + 1].sf


def test_explora_sf_sets_adr_method():
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
    explora_sf.apply(sim)
    assert sim.network_server.adr_method == "explora-sf"


def test_explora_sf_updates_with_new_node():
    random.seed(0)
    sim = Simulator(
        num_nodes=12,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=False,
        adr_server=False,
        seed=43,
    )
    explora_sf.apply(sim)
    gw = sim.gateways[0]
    rssis = [-40, -41, -55, -56, -70, -71, -85, -86, -100, -101, -115, -116]
    for i, (node, rssi) in enumerate(zip(sim.nodes, rssis)):
        frame = node.prepare_uplink(b"ping")
        sim.network_server.receive(i, node.id, gw.id, rssi, frame)
    for node in sim.nodes:
        dl = gw.pop_downlink(node.id)
        if dl is not None:
            node.handle_downlink(dl[0])
    sf_counts = {sf: 0 for sf in range(7, 13)}
    for node in sim.nodes:
        sf_counts[node.sf] += 1
    assert all(count == 2 for count in sf_counts.values())
    node_id = len(sim.nodes) + 1
    channel = sim.multichannel.select_mask(0xFFFF)
    new_node = Node(node_id, 0.0, 0.0, 12, 14.0, channel=channel, class_type=sim.node_class)
    new_node.initial_sf = 12
    new_node.initial_tx_power = 14.0
    new_node.channel.detection_threshold_dBm = (
        Channel.flora_detection_threshold(12, new_node.channel.bandwidth)
        + new_node.channel.sensitivity_margin_dB
    )
    sim.nodes.append(new_node)
    frame = new_node.prepare_uplink(b"ping")
    sim.network_server.receive(len(sim.nodes), new_node.id, gw.id, -60, frame)
    for node in sim.nodes:
        dl = gw.pop_downlink(node.id)
        if dl is not None:
            node.handle_downlink(dl[0])
    sf_counts = {sf: 0 for sf in range(7, 13)}
    for node in sim.nodes:
        sf_counts[node.sf] += 1
    assert sf_counts[7] == 3
    assert all(sf_counts[sf] == 2 for sf in range(8, 13))
    assert new_node.sf == sim.nodes[3].sf
