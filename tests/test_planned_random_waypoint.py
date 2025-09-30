from loraflexsim.launcher.planned_random_waypoint import PlannedRandomWaypoint
from loraflexsim.launcher.node import Node


def test_planner_avoids_high_obstacle():
    terrain = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    heights = [
        [0, 0, 0],
        [0, 10, 0],
        [0, 0, 0],
    ]
    mobility = PlannedRandomWaypoint(
        area_size=90.0,
        terrain=terrain,
        obstacle_height_map=heights,
        max_height=5,
    )
    node = Node(1, 0.0, 0.0, 7, 14)
    mobility.assign(node)
    cells = [mobility.planner._coord_to_cell(x, y) for x, y in node.path]
    assert (1, 1) not in cells


def test_assign_sets_path():
    terrain = [[1, 1], [1, 1]]
    mobility = PlannedRandomWaypoint(area_size=20.0, terrain=terrain)
    node = Node(1, 5.0, 5.0, 7, 14)
    mobility.assign(node)
    assert len(node.path) >= 2
