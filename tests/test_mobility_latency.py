import csv
from pathlib import Path

from loraflexsim.launcher import Simulator, MultiChannel


def test_mobility_latency() -> None:
    params = dict(
        num_nodes=5,
        area_size=100,
        transmission_mode="Random",
        packet_interval=0.1,
        packets_to_send=5,
        mobility=True,
        seed=1,
    )

    # Single-channel scenario
    sim_single = Simulator(**params, channels=MultiChannel([868_100_000.0]))
    sim_single.run()
    latency_single = sim_single.get_metrics()["avg_delay_s"]

    # Tri-channel scenario
    sim_multi = Simulator(
        **params,
        channels=MultiChannel([868_100_000.0, 868_300_000.0, 868_500_000.0]),
    )
    sim_multi.run()
    latency_multi = sim_multi.get_metrics()["avg_delay_s"]

    # Multi-channel should have lower latency
    assert latency_multi < latency_single

    # Export results for plotting
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "mobility_latency.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "avg_delay_s"])
        writer.writerow(["single_channel", latency_single])
        writer.writerow(["multi_channel", latency_multi])
