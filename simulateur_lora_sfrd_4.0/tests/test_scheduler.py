import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.server import NetworkServer  # noqa: E402


def test_scheduled_downlink_delivery():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    gw = Gateway(1, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]
    ns.send_downlink(node, b"data", at_time=5.0)

    ns.deliver_scheduled(node.id, 4.0)
    assert gw.pop_downlink(node.id) is None

    ns.deliver_scheduled(node.id, 5.0)
    frame = gw.pop_downlink(node.id)
    assert frame is not None
    assert isinstance(frame.payload, bytes) and frame.payload == b"data"
