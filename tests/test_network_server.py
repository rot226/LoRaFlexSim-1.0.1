import pytest

from loraflexsim.launcher.server import ADR_WINDOW_SIZE, NetworkServer
from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.node import Node
from loraflexsim.architecture import NetworkServer as SimpleServer



def test_deduplicate_packets():
    server = NetworkServer()
    gw1 = Gateway(0, 0, 0)
    gw2 = Gateway(1, 10, 0)
    node = Node(0, 0, 0, 7, 14)
    server.gateways = [gw1, gw2]
    server.nodes = [node]

    server.receive(1, node.id, gw1.id, -20, snir=5.0)
    server.receive(1, node.id, gw2.id, -22, snir=8.5)

    assert server.packets_received == 1
    assert server.duplicate_packets == 1
    assert server.event_gateway[1] == gw2.id
    assert pytest.approx(server.event_snir[1], abs=1e-9) == 8.5
    assert node.last_snr == 8.5
    assert node.gateway_snr_history.get(gw1.id, []) == []
    assert node.gateway_snr_history.get(gw2.id, []) == [8.5]


def test_add_gateway_deduplicates():
    server = SimpleServer()
    gw = Gateway(0, 0, 0)
    server.add_gateway(gw)
    server.add_gateway(gw)
    assert server.gateways == [gw]


def test_multi_gateway_adr_history_evolution():
    server = NetworkServer()
    gw1 = Gateway(0, 0, 0)
    gw2 = Gateway(1, 100, 0)
    node = Node(0, 0, 0, 7, 14)
    server.gateways = [gw1, gw2]
    server.nodes = [node]

    server.receive(1, node.id, gw1.id, snir=2.5)
    assert node.gateway_snr_history.get(gw1.id, []) == [2.5]
    assert node.gateway_snr_history.get(gw2.id, []) == []

    server.receive(2, node.id, gw2.id, snir=4.0)
    assert node.gateway_snr_history.get(gw1.id, []) == [2.5]
    assert node.gateway_snr_history.get(gw2.id, []) == [4.0]

    server.receive(3, node.id, gw1.id, snir=3.0)
    assert node.gateway_snr_history.get(gw1.id, []) == [2.5, 3.0]
    assert node.gateway_snr_history.get(gw2.id, []) == [4.0]

    server.receive(3, node.id, gw2.id, snir=5.0)
    assert node.gateway_snr_history.get(gw1.id, []) == [2.5]
    assert node.gateway_snr_history.get(gw2.id, []) == [4.0, 5.0]

    for event_id in range(4, 4 + ADR_WINDOW_SIZE):
        snr = float(event_id)
        server.receive(event_id, node.id, gw2.id, snir=snr)

    history_gw2 = node.gateway_snr_history.get(gw2.id, [])
    assert len(history_gw2) == ADR_WINDOW_SIZE
    assert history_gw2[-1] == float(3 + ADR_WINDOW_SIZE)
