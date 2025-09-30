from __future__ import annotations

import csv
from pathlib import Path

from loraflexsim.launcher import MultiChannel, Simulator

PARAMS = dict(
    num_nodes=20,
    area_size=100,
    transmission_mode="Random",
    packet_interval=1.0,
    packets_to_send=10,
    mobility=True,
    seed=1,
    fixed_sf=7,
    battery_capacity_j=1000,
)


def run_sim(channels: MultiChannel) -> tuple[float, int]:
    sim = Simulator(**PARAMS, channels=channels)
    sim.run()
    metrics = sim.get_metrics()
    energy = metrics["energy_J"]
    delivered = sim.packets_delivered
    return energy, delivered


def test_mobility_energy_per_packet() -> None:
    single_channels = MultiChannel([868_100_000.0])
    multi_channels = MultiChannel([868_100_000.0, 868_300_000.0, 868_500_000.0])

    energy_single, delivered_single = run_sim(single_channels)
    energy_multi, delivered_multi = run_sim(multi_channels)

    assert delivered_single > 0 and delivered_multi > 0

    energy_per_packet_single = energy_single / delivered_single
    energy_per_packet_multi = energy_multi / delivered_multi

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "mobility_energy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "energy_J", "packets_delivered", "energy_per_packet"])
        writer.writerow([
            "single_channel",
            energy_single,
            delivered_single,
            energy_per_packet_single,
        ])
        writer.writerow([
            "multi_channel",
            energy_multi,
            delivered_multi,
            energy_per_packet_multi,
        ])

    assert energy_per_packet_multi < energy_per_packet_single
