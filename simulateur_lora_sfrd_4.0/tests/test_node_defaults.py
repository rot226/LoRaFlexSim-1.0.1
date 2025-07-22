import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402


def test_node_default_channel():
    node = Node(1, 0.0, 0.0, 7, 14.0)
    # A default Channel instance should be created
    assert node.channel is not None
    # The channel should provide airtime calculation without error
    assert node.channel.airtime(7) > 0


def test_distance_to_uses_altitude():
    node = Node(1, 0.0, 0.0, 7, 14.0)
    gw = Gateway(1, 0.0, 0.0, altitude=10.0)
    node.altitude = 0.0
    assert node.distance_to(gw) == 10.0
