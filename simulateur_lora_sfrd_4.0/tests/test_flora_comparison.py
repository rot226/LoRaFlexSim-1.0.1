import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.simulator import Simulator  # noqa: E402
from VERSION_4.launcher.compare_flora import load_flora_metrics, compare_with_sim  # noqa: E402


def _make_sim(num_nodes: int) -> Simulator:
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    gw = sim.gateways[0]
    for idx, node in enumerate(sim.nodes):
        node.x = gw.x
        node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    for idx, node in enumerate(sim.nodes):
        sim.schedule_event(node, idx * 1.0)
    return sim


def _make_colliding_sim() -> Simulator:
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=2,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    gw = sim.gateways[0]
    for node in sim.nodes:
        node.x = gw.x
        node.y = gw.y
    sim.event_queue.clear()
    sim.event_id_counter = 0
    for node in sim.nodes:
        sim.schedule_event(node, 0.0)
    return sim


@pytest.mark.parametrize(
    "csv_file,num_nodes",
    [
        ("flora_full.csv", 2),
        ("flora_sample.csv", 2),
        ("flora_three.csv", 3),
    ],
)
def test_compare_with_flora_sample(csv_file: str, num_nodes: int) -> None:
    """Simulator metrics should match the provided FLoRa references."""
    sim = _make_sim(num_nodes=num_nodes)
    sim.run()
    metrics = sim.get_metrics()
    flora_csv = Path(__file__).parent / "data" / csv_file
    flora_metrics = load_flora_metrics(flora_csv)

    assert metrics["PDR"] == pytest.approx(flora_metrics["PDR"], abs=0.01)
    assert metrics["sf_distribution"] == flora_metrics["sf_distribution"]
    assert compare_with_sim(metrics, flora_csv)


def test_collision_distribution_against_flora():
    sim = _make_colliding_sim()
    while sim.step():
        pass
    flora_csv = Path(__file__).parent / "data" / "flora_collisions.csv"
    flora_metrics = load_flora_metrics(flora_csv)
    sim_cd = {7: sum(n.packets_collision for n in sim.nodes if n.sf == 7)}
    assert sim.packets_lost_collision == flora_metrics["collisions"]
    assert sim.packets_lost_collision == sum(flora_metrics["collision_distribution"].values())
    assert sim_cd == flora_metrics["collision_distribution"]
