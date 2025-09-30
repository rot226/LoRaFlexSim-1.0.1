import pytest

scipy = pytest.importorskip("scipy")
from scipy.stats import kstest, expon

from loraflexsim.launcher.simulator import Simulator


def test_poisson_interval_kstest():
    mean_interval = 5.0
    packets = 1000
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=mean_interval,
        packets_to_send=packets,
        pure_poisson_mode=True,
        mobility=False,
        seed=42,
    )
    sim.run()
    node = sim.nodes[0]
    times = [entry["tx_time"] for entry in node.interval_log]
    intervals = [t1 - t0 for t0, t1 in zip(times, times[1:])]
    stat, p = kstest(intervals, expon(scale=mean_interval).cdf)
    assert p >= 0.05
