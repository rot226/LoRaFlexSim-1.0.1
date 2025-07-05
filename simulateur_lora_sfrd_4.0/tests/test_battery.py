import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402


def test_battery_level_zero_capacity():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), battery_capacity_j=0)
    assert node.battery_level == 0.0

