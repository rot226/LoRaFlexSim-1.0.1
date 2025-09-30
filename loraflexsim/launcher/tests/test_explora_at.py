import random
import pytest

from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher import explora_at, server


def test_explora_at_enables_adr_and_sets_method():
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
    explora_at.apply(sim)
    assert sim.adr_node is True
    assert sim.adr_server is True
    assert sim.network_server.adr_method == "explora-at"


def test_explora_at_initialisation_margin_and_sf():
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
    explora_at.apply(sim)
    node = sim.nodes[0]
    assert Simulator.MARGIN_DB == 10.0
    assert server.MARGIN_DB == 10.0
    assert node.sf == node.initial_sf == 12
    expected_thr = Channel.flora_detection_threshold(12, node.channel.bandwidth) + node.channel.sensitivity_margin_dB
    assert node.channel.detection_threshold_dBm == expected_thr


def test_explora_at_uniform_airtime_groups():
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
    explora_at.apply(sim)
    gw = sim.gateways[0]
    rssis = [-40, -41, -55, -56, -70, -71, -85, -86, -100, -101, -115, -116]
    for i, (node, rssi) in enumerate(zip(sim.nodes, rssis)):
        frame = node.prepare_uplink(b"ping")
        sim.network_server.receive(i, node.id, gw.id, rssi, frame)
    counts = {sf: 0 for sf in range(7, 13)}
    for node in sim.nodes:
        dl = gw.pop_downlink(node.id)
        if dl is not None:
            node.handle_downlink(dl[0])
        counts[node.sf] += 1
    assert counts[7] == 6
    assert counts[8] == 3
    assert counts[9] == 2
    assert counts[10] == 1
    assert counts[11] == 0
    assert counts[12] == 0


@pytest.mark.parametrize(
    "payload_sizes",
    [
        [20] * 13,
        [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 40],
    ],
)
def test_explora_at_airtime_balancing_varied_snr(payload_sizes):
    random.seed(0)
    num_nodes = len(payload_sizes)
    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        duty_cycle=None,
        mobility=False,
        adr_node=False,
        adr_server=False,
        seed=123,
    )
    explora_at.apply(sim)
    gw = sim.gateways[0]
    rssis = [-40 - 5 * i for i in range(num_nodes)]
    for i, (node, rssi) in enumerate(zip(sim.nodes, rssis)):
        frame = node.prepare_uplink(b"0" * payload_sizes[i])
        sim.network_server.receive(i, node.id, gw.id, rssi, frame)
    for node in sim.nodes:
        dl = gw.pop_downlink(node.id)
        if dl is not None:
            node.handle_downlink(dl[0])

    # Actual distribution across SFs
    counts = {sf: 0 for sf in range(7, 13)}
    for node in sim.nodes:
        counts[node.sf] += 1

    # Expected distribution derived from airtime-fair algorithm
    ch = sim.channel
    airtimes = {
        sf: ch.airtime(sf, Simulator.EXPLORA_AT_PAYLOAD_SIZE) for sf in range(7, 13)
    }
    inv = {sf: 1.0 / t for sf, t in airtimes.items()}
    total = sum(inv.values())
    raw = {sf: num_nodes * inv[sf] / total for sf in range(7, 13)}
    expected = {sf: int(raw[sf]) for sf in range(7, 13)}
    remaining = num_nodes - sum(expected.values())
    order = sorted(range(7, 13), key=lambda s: (-(raw[s] - expected[s]), s))
    for sf in order[:remaining]:
        expected[sf] += 1
    assert counts == expected

    # Compute total airtime per SF group using provided payload sizes
    totals = {sf: 0.0 for sf in range(7, 13)}
    for node, size in zip(sim.nodes, payload_sizes):
        totals[node.sf] += node.channel.airtime(node.sf, payload_size=size)

    # Groups with more than one node should have approximately equal airtime
    non_zero = [totals[sf] for sf in range(7, 13) if counts[sf] > 1]
    if non_zero:
        avg = sum(non_zero) / len(non_zero)
        for total in non_zero:
            assert total == pytest.approx(avg, rel=0.2)
