import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.server import NetworkServer  # noqa: E402
from VERSION_4.launcher.lorawan import JoinRequest, JoinAccept  # noqa: E402


def test_otaa_join_procedure():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), activated=False, appkey=bytes(16))
    gw = Gateway(1, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]

    req = node.prepare_uplink(b"")
    assert isinstance(req, JoinRequest)

    ns._activate(node)
    frame = gw.pop_downlink(node.id)
    assert isinstance(frame, JoinAccept)

    node.handle_downlink(frame)
    assert node.activated
    assert len(node.nwkskey) == 16
    assert len(node.appskey) == 16
    assert node.downlink_pending == 0
