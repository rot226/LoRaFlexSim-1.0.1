import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.path_mobility import PathMobility  # noqa: E402
from VERSION_4.launcher.simulator import Simulator  # noqa: E402


def test_path_mobility_avoids_center_obstacle():
    grid = [
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    mob = PathMobility(area_size=90.0, path_map=grid, min_speed=10.0, max_speed=10.0)
    node = Node(1, 10.0, 45.0, 7, 14.0, channel=Channel())
    mob.assign(node)
    cells = [mob._coord_to_cell(x, y) for x, y in node.path]
    assert (1, 1) not in cells
    first = (node.x, node.y)
    mob.move(node, 1.0)
    assert (node.x, node.y) != first


def test_simulator_uses_path_mobility(tmp_path):
    path_file = tmp_path / "map.json"
    path_file.write_text("[[0,0],[0,0]]")
    sim = Simulator(num_nodes=1, num_gateways=1, area_size=100.0, mobility=True,
                    path_map=str(path_file))
    assert isinstance(sim.mobility_model, PathMobility)

