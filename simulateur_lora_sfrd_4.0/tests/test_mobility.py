import sys
from pathlib import Path
import random
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.smooth_mobility import SmoothMobility  # noqa: E402


def test_smooth_mobility_progress_and_new_path():
    random.seed(0)
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    mobility = SmoothMobility(area_size=100.0, min_speed=10.0, max_speed=10.0)

    mobility.assign(node)
    initial_path = node.path
    initial_duration = node.path_duration

    half_time = initial_duration * 0.5
    mobility.move(node, half_time)

    assert node.path == initial_path
    assert node.path_progress == pytest.approx(0.5)
    first_pos = (node.x, node.y)
    assert first_pos != (0.0, 0.0)

    next_time = initial_duration * 1.2
    mobility.move(node, next_time)

    assert node.path != initial_path
    assert node.path_progress == pytest.approx(0.2)
    assert node.path_duration != initial_duration
    second_pos = (node.x, node.y)
    assert second_pos != first_pos

    prev_progress = node.path_progress
    mobility.move(node, next_time + node.path_duration * 0.5)

    assert node.path_progress > prev_progress
    assert (node.x, node.y) != second_pos
