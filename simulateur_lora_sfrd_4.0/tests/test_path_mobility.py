import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.path_mobility import PathMapMobility  # noqa: E402


def test_path_mobility_avoids_obstacles():
    grid = [
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    mob = PathMapMobility(grid, area_size=3.0, min_speed=1.0, max_speed=1.0)
    start_node = Node(1, 0.5, 0.5, 7, 14.0, channel=Channel())
    dest = (2.5, 0.5)
    path = mob.compute_path(start_node.x, start_node.y, dest=dest)
    # Convert path points to grid cells
    cells = [mob._cell_from_pos(x, y) for x, y in path]
    # Ensure obstacle cell (1,0) is not in the path
    assert (1, 0) not in cells
    assert cells[-1] == mob._cell_from_pos(*dest)

