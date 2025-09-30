import json
from loraflexsim.launcher.path_mobility import PathMobility
from loraflexsim.launcher.node import Node

def test_dynamic_obstacle_blocks_move(tmp_path):
    path_map = [[0, 0], [0, 0]]
    obstacle = [{"x": 50, "y": 50, "radius": 5, "vx": 0, "vy": 0}]
    obs_file = tmp_path / "obs.json"
    obs_file.write_text(json.dumps(obstacle))
    mobility = PathMobility(100.0, path_map, dynamic_obstacles=str(obs_file))
    node = Node(1, 40.0, 50.0, 7, 14)
    mobility.assign(node)
    node.path = [(40.0, 50.0), (50.0, 50.0)]
    node.speed = 10.0
    mobility.move(node, 2.0)
    assert (node.x, node.y) == (40.0, 50.0)


def test_find_path_avoids_negative_cell():
    path_map = [
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0],
    ]
    mobility = PathMobility(100.0, path_map)
    path = mobility._find_path((0, 1), (2, 1))
    cells = [mobility._coord_to_cell(x, y) for x, y in path]
    assert (1, 1) not in cells


def test_find_path_avoids_high_obstacle():
    path_map = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    height = [
        [0, 10, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    mobility = PathMobility(100.0, path_map, obstacle_height_map=height, max_height=5)
    path = mobility._find_path((0, 0), (2, 0))
    cells = [mobility._coord_to_cell(x, y) for x, y in path]
    assert (1, 0) not in cells

