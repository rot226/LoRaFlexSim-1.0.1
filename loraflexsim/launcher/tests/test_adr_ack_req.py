import random
from loraflexsim.launcher.server import NetworkServer
from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.node import Node


def test_adr_ack_req_resets_counter():
    random.seed(0)
    server = NetworkServer()
    gw = Gateway(0, 0, 0)
    server.gateways = [gw]
    node = Node(0, 0, 0, 7, 14)
    server.nodes = [node]

    # simulate adr_ack_limit + 1 uplinks with server processing
    for i in range(node.adr_ack_limit + 1):
        frame = node.prepare_uplink(b"ping")
        server.receive(i, node.id, gw.id, -40, frame)
        dl = gw.pop_downlink(node.id)
        if dl:
            node.handle_downlink(dl[0])

    assert node.adr_ack_cnt == 0
    assert node.sf == 7

    # adr_ack_delay more uplinks should not trigger ADR fallback
    for i in range(node.adr_ack_delay):
        frame = node.prepare_uplink(b"ping")
        server.receive(100 + i, node.id, gw.id, -40, frame)

    assert node.sf == 7
    assert node.adr_ack_cnt == node.adr_ack_delay
