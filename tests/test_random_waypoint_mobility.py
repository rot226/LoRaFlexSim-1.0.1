from loraflexsim.launcher.random_waypoint import RandomWaypoint
from loraflexsim.launcher.node import Node


def test_random_waypoint_move_within_bounds_and_updates_time():
    mobility = RandomWaypoint(area_size=100.0, min_speed=1.0, max_speed=1.0)
    node = Node(0, 50.0, 50.0, 7, 14)

    mobility.assign(node)
    mobility.move(node, 10.0)

    assert 0.0 <= node.x <= 100.0
    assert 0.0 <= node.y <= 100.0
    assert node.last_move_time == 10.0
