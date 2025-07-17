import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.simulator import Simulator  # noqa: E402


def test_battery_level_zero_capacity():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), battery_capacity_j=0)
    assert node.battery_level == 0.0


def test_battery_consumption_decreases_level():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), battery_capacity_j=10)
    assert node.battery_level == 1.0
    node.add_energy(1.5, state="tx")
    assert node.battery_remaining_j == 8.5
    assert round(node.battery_level, 2) == 0.85


def _consumption_for_power(tx_power: float):
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=tx_power,
        battery_capacity_j=10.0,
    )
    node = sim.nodes[0]
    gw = sim.gateways[0]
    node.x = gw.x
    node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    sim.schedule_event(node, 0.0)
    sim.run()
    consumed = 10.0 - node.battery_remaining_j
    duration = node.channel.airtime(7, payload_size=sim.payload_size_bytes)
    profile = node.profile
    expected = profile.get_tx_current(tx_power) * profile.voltage_v * duration
    return consumed, expected


def test_tx_power_mapping_affects_consumption():
    low, expected_low = _consumption_for_power(2.0)
    high, expected_high = _consumption_for_power(14.0)
    assert high > low
    assert low == pytest.approx(expected_low, rel=1e-3)
    assert high == pytest.approx(expected_high, rel=1e-3)

