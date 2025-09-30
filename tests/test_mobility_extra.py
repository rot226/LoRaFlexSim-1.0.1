from loraflexsim.launcher.path_mobility import PathMobility
from loraflexsim.launcher.terrain_mobility import TerrainMapMobility
from loraflexsim.launcher.node import Node


def test_path_mobility_altitude_update():
    path_map = [[0, 0], [0, 0]]
    elevation = [[0, 10], [0, 0]]
    mobility = PathMobility(area_size=100.0, path_map=path_map, elevation=elevation)
    node = Node(1, 0.0, 0.0, 7, 14)
    start = mobility._cell_to_coord((0, 0))
    dest = mobility._cell_to_coord((1, 0))
    node.x, node.y = start
    node.path = [start, dest]
    node.speed = 50.0
    mobility.move(node, 1.0)
    assert node.altitude == 10.0
    assert (node.x, node.y) == dest


def test_terrain_mobility_speed_factor():
    terrain = [[1, 1]]
    mobility = TerrainMapMobility(area_size=100.0, terrain=terrain)
    node = Node(1, 0.0, 50.0, 7, 14)
    start = mobility._cell_to_coord((0, 0))
    dest = mobility._cell_to_coord((1, 0))
    node.x, node.y = start
    node.path = [start, dest]
    node.speed = 10.0
    mobility.move(node, 1.0)
    assert node.x == 35.0
    assert node.y == start[1]

