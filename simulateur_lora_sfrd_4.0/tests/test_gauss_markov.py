import sys
from pathlib import Path
import random
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.gauss_markov import GaussMarkov  # noqa: E402


def test_gauss_markov_moves():
    random.seed(0)
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    mob = GaussMarkov(area_size=100.0, mean_speed=5.0, alpha=0.5, step=1.0)
    mob.assign(node)
    first = (node.x, node.y)
    mob.move(node, 1.0)
    assert node.last_move_time == pytest.approx(1.0)
    assert (node.x, node.y) != first
