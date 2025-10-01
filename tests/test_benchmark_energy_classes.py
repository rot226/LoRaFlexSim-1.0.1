from pathlib import Path
import csv

from loraflexsim.launcher.simulator import Simulator
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
