import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.simulator import Simulator  # noqa: E402
from VERSION_4.launcher.compare_flora import (
    load_flora_metrics,
    compare_with_sim,
)  # noqa: E402


def _make_sim(num_nodes: int, *, same_start: bool = False) -> Simulator:
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
        sim.schedule_event(node, 0.0 if same_start else idx * 1.0)
    return sim


def test_compare_with_flora_sample():
    sim = _make_sim(num_nodes=2)
    sim.run()
    metrics = sim.get_metrics()
    flora_csv = Path(__file__).parent / "data" / "flora_sample.csv"
    flora_metrics = load_flora_metrics(flora_csv)
    assert metrics["PDR"] == pytest.approx(flora_metrics["PDR"])
    assert compare_with_sim(metrics, flora_csv)


def _add_collision_distribution(sim: Simulator, metrics: dict) -> None:
    """Helper to compute the collision distribution per SF."""
    coll_dist = {}
    for sf in range(7, 13):
        coll_dist[sf] = sum(
            n.packets_collision for n in sim.nodes if n.sf == sf
        )
    metrics["collisions_distribution"] = coll_dist


def test_compare_with_extended_metrics():
    sim = _make_sim(num_nodes=2)
    sim.run()
    metrics = sim.get_metrics()
    _add_collision_distribution(sim, metrics)
    flora_csv = Path(__file__).parent / "data" / "flora_extended.csv"
    flora_metrics = load_flora_metrics(flora_csv)
    assert metrics["throughput_bps"] == pytest.approx(
        flora_metrics["throughput_bps"]
    )
    assert metrics["energy_J"] == pytest.approx(flora_metrics["energy_J"])
    assert compare_with_sim(
        metrics,
        flora_csv,
        pdr_tol=1e-6,
        throughput_tol=1e-6,
        energy_tol=1e-6,
    )


def test_compare_with_collision_metrics():
    sim = _make_sim(num_nodes=3, same_start=True)
    sim.run()
    metrics = sim.get_metrics()
    _add_collision_distribution(sim, metrics)
    flora_csv = Path(__file__).parent / "data" / "flora_collision.csv"
    assert compare_with_sim(
        metrics,
        flora_csv,
        pdr_tol=1e-6,
        throughput_tol=1e-6,
        energy_tol=1e-6,
    )
