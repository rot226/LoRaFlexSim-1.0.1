import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel
from VERSION_4.launcher.simulator import Simulator
from VERSION_4.launcher.compare_flora import load_flora_metrics, compare_with_sim


def _make_sim(num_nodes: int) -> Simulator:
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=num_nodes,
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


def test_compare_with_flora_sample():
    sim = _make_sim(num_nodes=2)
    sim.run()
    metrics = sim.get_metrics()
    flora_csv = Path(__file__).parent / "data" / "flora_sample.csv"
    flora_metrics = load_flora_metrics(flora_csv)
    assert metrics["PDR"] == pytest.approx(flora_metrics["PDR"])
    assert compare_with_sim(metrics, flora_csv)
