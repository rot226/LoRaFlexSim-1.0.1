import math

from loraflexsim.launcher.energy_profiles import EnergyProfile
from loraflexsim.launcher.node import Node


def make_node(class_type: str) -> Node:
    profile = EnergyProfile(startup_time_s=0.0)
    return Node(
        node_id=1,
        x=0.0,
        y=0.0,
        sf=7,
        tx_power=14.0,
        class_type=class_type,
        energy_profile=profile,
    )


def test_class_c_starts_in_rx():
    node = make_node("C")
    assert node.state == "rx"


def test_class_b_defaults_to_sleep():
    node = make_node("B")
    assert node.state == "sleep"


def test_distance_to_supports_altitude():
    node = make_node("A")
    other = make_node("A")
    other.x = 3.0
    other.y = 4.0
    other.altitude = 12.0
    node.altitude = 8.0
    assert math.isclose(node.distance_to(other), math.sqrt(41))
