from pathlib import Path
import csv

import pytest

from loraflexsim.launcher.simulator import EventType, Simulator
from scripts import benchmark_energy_classes


def test_benchmark_energy_classes(tmp_path) -> None:
    out = tmp_path / "energy.csv"
    result = benchmark_energy_classes.main(
        [
            "--nodes",
            "1",
            "--packets",
            "1",
            "--interval",
            "1.0",
            "--output",
            str(out),
            "--mode",
            "Periodic",
            "--seed",
            "2",
            "--duty-cycle",
            "0.0",
        ]
    )
    assert Path(result) == out
    assert out.exists()
    with out.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert [row["class"] for row in rows] == ["A", "B", "C"]
    for row in rows:
        total = float(row["energy_nodes_J"])
        assert total >= 0.0
        assert float(row["energy_per_node_J"]) >= 0.0
        energy_states = [
            float(value)
            for key, value in row.items()
            if key.startswith("energy_")
            and key not in {"energy_nodes_J", "energy_per_node_J"}
        ]
        assert energy_states
        assert abs(sum(energy_states) - total) <= 1e-6 + 1e-3 * total


def test_energy_cleanup_for_limited_packets() -> None:
    base_kwargs = dict(
        num_nodes=1,
        num_gateways=1,
        area_size=100.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        duty_cycle=None,
        mobility=False,
        seed=123,
    )

    sim_b = Simulator(node_class="B", **base_kwargs)
    sim_b.run()
    metrics_b = sim_b.get_metrics()

    sim_c = Simulator(node_class="C", **base_kwargs)
    sim_c.run()
    metrics_c = sim_c.get_metrics()

    energy_b = metrics_b["energy_nodes_J"] / base_kwargs["num_nodes"]
    energy_c = metrics_c["energy_nodes_J"] / base_kwargs["num_nodes"]
    assert energy_b <= energy_c * 1.5

    node_c = sim_c.nodes[0]
    breakdown_c = metrics_c["energy_breakdown_by_node"][node_c.id]
    rx_energy = breakdown_c.get("listen", 0.0) + breakdown_c.get("rx", 0.0)
    current = (
        node_c.profile.listen_current_a
        if node_c.profile.listen_current_a > 0.0
        else node_c.profile.rx_current_a
    )
    min_rx_energy = (
        2 * current * node_c.profile.voltage_v * node_c.profile.rx_window_duration
    )
    tolerance = max(1e-9, min_rx_energy * 0.05)
    assert rx_energy + tolerance >= min_rx_energy


@pytest.mark.slow
def test_class_energy_ordering_matches_flora_profile() -> None:
    """Class B/C listening cost should dominate energy consumption."""

    base_kwargs = dict(
        num_nodes=5,
        num_gateways=1,
        area_size=1000.0,
        transmission_mode="Periodic",
        packet_interval=60.0,
        packets_to_send=3,
        duty_cycle=None,
        mobility=False,
        flora_mode=True,
        flora_timing=True,
        seed=4,
    )

    per_node_energy: dict[str, float] = {}
    for cls in ("A", "B", "C"):
        sim = Simulator(node_class=cls, **base_kwargs)
        sim.run()
        metrics = sim.get_metrics()
        per_node_energy[cls] = (
            metrics["energy_nodes_J"] / base_kwargs["num_nodes"]
            if base_kwargs["num_nodes"] > 0
            else 0.0
        )

    assert per_node_energy["A"] * 2 < per_node_energy["B"]
    assert per_node_energy["B"] * 10 < per_node_energy["C"]


def test_class_b_beacon_listen_energy_accounting() -> None:
    base_kwargs = dict(
        num_nodes=1,
        num_gateways=1,
        area_size=100.0,
        transmission_mode="Periodic",
        packet_interval=60.0,
        packets_to_send=0,
        duty_cycle=None,
        mobility=False,
        flora_mode=True,
        flora_timing=True,
        seed=1,
    )

    sim = Simulator(node_class="B", **base_kwargs)
    node = sim.nodes[0]

    while sim.event_queue and sim.event_queue[0].type != EventType.BEACON:
        sim.step()

    metrics_before = sim.get_metrics()
    energy_before = metrics_before["energy_nodes_J"]

    sim.step()  # Traiter le beacon courant

    metrics_after = sim.get_metrics()
    energy_after = metrics_after["energy_nodes_J"]

    assert energy_after > energy_before

    state = "listen" if node.profile.listen_current_a > 0.0 else "rx"
    duration = getattr(node.profile, "beacon_listen_duration", 0.0)
    if duration <= 0.0:
        duration = node.profile.rx_window_duration
    expected = node.profile.energy_for(state, duration)

    assert energy_after == pytest.approx(
        energy_before + expected, rel=0.05, abs=1e-9
    )

    breakdown_before = metrics_before["energy_breakdown_by_node"][node.id]
    breakdown_after = metrics_after["energy_breakdown_by_node"][node.id]
    delta_breakdown = breakdown_after.get(state, 0.0) - breakdown_before.get(state, 0.0)
    assert delta_breakdown == pytest.approx(expected, rel=0.05, abs=1e-9)

    assert node.last_state_time == pytest.approx(duration)
    assert node.state == "sleep"
