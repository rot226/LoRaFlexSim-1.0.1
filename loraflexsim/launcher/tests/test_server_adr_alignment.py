"""Regression tests for ADR alignment between client and server."""

from __future__ import annotations

from loraflexsim.launcher.server import NetworkServer
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.lorawan import LoRaWANFrame, LinkADRReq, LinkADRAns


def _make_node(node_id: int, sf: int, power: float) -> Node:
    node = Node(node_id, 0.0, 0.0, sf, power, security=False)
    node.adr = True
    node.activated = True
    node.security_enabled = False
    node.last_uplink_end_time = 1.0
    node.frames_since_last_adr_command = 0
    return node


def test_server_uses_margin_and_rounding_for_positive_steps(monkeypatch):
    server = NetworkServer()
    server.adr_enabled = True
    server.adr_method = "avg"
    server.MARGIN_DB = 10.0

    gw = Gateway(0, 0.0, 0.0)

    node = _make_node(1, 9, 14.0)
    node.snr_history = [(gw.id, 5.0)] * 19
    node.frames_since_last_adr_command = 19
    server.nodes = [node]
    server.gateways = [gw]

    captured: dict[str, tuple] = {}

    def fake_send_downlink(target, payload=b"", confirmed=False, adr_command=None, **kwargs):
        captured["adr_command"] = adr_command

    monkeypatch.setattr(server, "send_downlink", fake_send_downlink)

    server.receive(1, node.id, gw.id, rssi=-120.0, frame=None, end_time=1.0, snir=5.0)

    assert captured["adr_command"] == (7, 12.0, node.chmask, node.nb_trans)


def test_server_rounding_matches_flora_for_negative_steps(monkeypatch):
    server = NetworkServer()
    server.adr_enabled = True
    server.adr_method = "avg"
    server.MARGIN_DB = 10.0

    gw = Gateway(1, 0.0, 0.0)

    node = _make_node(2, 9, 8.0)
    node.snr_history = [(gw.id, -10.0)] * 19
    node.frames_since_last_adr_command = 19
    server.nodes = [node]
    server.gateways = [gw]

    captured: dict[str, tuple] = {}

    def fake_send_downlink(target, payload=b"", confirmed=False, adr_command=None, **kwargs):
        captured["adr_command"] = adr_command

    monkeypatch.setattr(server, "send_downlink", fake_send_downlink)

    server.receive(1, node.id, gw.id, rssi=-130.0, frame=None, end_time=1.0, snir=-10.0)

    assert captured["adr_command"] == (9, 14.0, node.chmask, node.nb_trans)


def test_node_resets_adr_ack_counter_on_downlink():
    node = _make_node(3, 9, 14.0)
    node.adr_ack_limit = 2
    node.adr_ack_delay = 1

    node.prepare_uplink(b"")
    node.prepare_uplink(b"")
    assert node.adr_ack_cnt == 2

    req = LinkADRReq(5, 0)
    frame = LoRaWANFrame(mhdr=0x60, fctrl=0, fcnt=0, payload=req.to_bytes())
    node.handle_downlink(frame)

    assert node.adr_ack_cnt == 0
    assert node.pending_mac_cmd == LinkADRAns().to_bytes()
