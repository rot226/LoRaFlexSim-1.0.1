import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402


def test_battery_level_zero_capacity():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), battery_capacity_j=0)
    assert node.battery_level == 0.0


def test_battery_consumption_decreases_level():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), battery_capacity_j=10)
    assert node.battery_level == 1.0
    node.add_energy(1.5, state="tx")
    assert node.battery_remaining_j == 8.5
    assert round(node.battery_level, 2) == 0.85

