import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.server import NetworkServer  # noqa: E402
from VERSION_4.launcher.lorawan import LoRaWANFrame, JoinAccept  # noqa: E402


def test_send_downlink_immediate():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(shadowing_std=0))
    gw = Gateway(1, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]

    ns.send_downlink(node, b"data", confirmed=True, request_ack=True)
    frame = gw.pop_downlink(node.id)
    assert isinstance(frame, LoRaWANFrame)
    assert frame.confirmed
    assert frame.fctrl == 0x20
    assert node.fcnt_down == 1
    assert node.downlink_pending == 1


def test_receive_triggers_otaa_activation():
    random.seed(0)
    node = Node(2, 0.0, 0.0, 7, 14.0, channel=Channel(shadowing_std=0), activated=False, appkey=bytes(16))
    gw = Gateway(2, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]
    ns.channel = Channel(shadowing_std=0)

    ns.receive(event_id=1, node_id=node.id, gateway_id=gw.id, rssi=-120, frame=None)
    frame = gw.pop_downlink(node.id)
    assert isinstance(frame, JoinAccept)
    assert len(node.nwkskey) == 16
    assert len(node.appskey) == 16
    assert ns.next_devaddr == 2
    assert node.fcnt_down == 1


def test_receive_decrypts_uplink():
    node = Node(3, 0.0, 0.0, 7, 14.0, channel=Channel(shadowing_std=0))
    gw = Gateway(3, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]

    frame = node.prepare_uplink(b"hello")
    rx = LoRaWANFrame(
        mhdr=frame.mhdr,
        fctrl=frame.fctrl,
        fcnt=frame.fcnt,
        payload=b"",
        confirmed=frame.confirmed,
        mic=frame.mic,
        encrypted_payload=frame.encrypted_payload,
    )

    ns.receive(event_id=2, node_id=node.id, gateway_id=gw.id, rssi=-120, frame=rx)
    assert rx.payload == b"hello"
