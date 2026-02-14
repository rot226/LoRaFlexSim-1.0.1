import pytest
from loraflexsim.launcher.simulator import Simulator


def test_qos_refresh_benchmark_includes_total_and_max_costs():
    simulator = Simulator(num_nodes=0, num_gateways=1, mobility=False, seed=1)
    simulator._qos_refresh_count = 3
    simulator._qos_refresh_total_cost_s = 1.2
    simulator._qos_refresh_max_cost_s = 0.7

    metrics = simulator.get_metrics()
    benchmark = metrics["qos_refresh_benchmark"]

    assert benchmark["refresh_count"] == 3
    assert benchmark["total_refresh_cost_s"] == 1.2
    assert benchmark["max_refresh_cost_s"] == 0.7
    assert benchmark["avg_refresh_cost_s"] == pytest.approx(0.4)
