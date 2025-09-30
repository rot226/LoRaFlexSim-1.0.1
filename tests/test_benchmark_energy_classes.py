from pathlib import Path
import csv

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
