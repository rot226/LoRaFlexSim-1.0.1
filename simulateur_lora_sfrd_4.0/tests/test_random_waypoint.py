import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.mobility import RandomWaypoint  # noqa: E402


def test_random_waypoint_blocked_cell():
    terrain = [[1.0, -1.0], [1.0, 1.0]]
    mob = RandomWaypoint(area_size=100.0, min_speed=10.0, max_speed=10.0, terrain=terrain)
    node = Node(1, 40.0, 10.0, 7, 14.0, channel=Channel())
    node.vx = 10.0
    node.vy = 0.0
    node.last_move_time = 0.0
    mob.move(node, 1.0)
    assert node.x == pytest.approx(40.0)
    assert node.vx == pytest.approx(-10.0)
