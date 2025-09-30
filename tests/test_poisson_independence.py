from loraflexsim.launcher.simulator import Simulator
import pytest
import statistics
from scipy.stats import pearsonr


def test_duty_cycle_keeps_mean_interval():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=5.0,
        packets_to_send=30,
        duty_cycle=0.01,
        mobility=False,
        pure_poisson_mode=True,
        seed=123,
    )
    sim.run()
    metrics = sim.get_metrics()
    assert abs(metrics["avg_arrival_interval_s"] - 5.0) / 5.0 < 0.15


def test_collisions_keep_mean_interval():
    sim = Simulator(
        num_nodes=3,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=5.0,
        packets_to_send=200,
        duty_cycle=0.01,
        mobility=False,
        pure_poisson_mode=True,
        seed=42,
    )
    sim.run()
    metrics = sim.get_metrics()
    assert abs(metrics["avg_arrival_interval_s"] - 5.0) / 5.0 < 0.15


@pytest.mark.parametrize("packet_interval", [10.0, 60.0, 600.0])
def test_poisson_process_independence(packet_interval: float) -> None:
    sim = Simulator(
        num_nodes=2,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=packet_interval,
        packets_to_send=5000,
        duty_cycle=0.01,
        mobility=False,
        pure_poisson_mode=True,
        seed=1,
    )
    sim.run()

    intervals = []
    for node in sim.nodes:
        times = [entry["poisson_time"] for entry in node.interval_log]
        deltas = [t1 - t0 for t0, t1 in zip(times, times[1:])]
        mean = statistics.fmean(deltas)
        assert abs(mean - packet_interval) / packet_interval <= 0.02
        std = statistics.pstdev(deltas)
        cv = std / mean if mean else 0.0
        assert abs(cv - 1.0) <= 0.02
        intervals.append(deltas)

    n = min(len(intervals[0]), len(intervals[1]))
    corr, p = pearsonr(intervals[0][:n], intervals[1][:n])
    assert abs(corr) < 0.05
    assert p >= 0.05
