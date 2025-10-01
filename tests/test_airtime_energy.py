from loraflexsim.launcher.simulator import Simulator
import pytest


def test_energy_and_airtime_metrics():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=2,
        mobility=False,
        seed=0,
    )
    sim.run()
    metrics = sim.get_metrics()
    node = sim.nodes[0]
    nid = node.id
    assert abs(metrics["energy_by_node"][nid] - node.energy_consumed) < 1e-9
    assert abs(metrics["airtime_by_node"][nid] - node.total_airtime) < 1e-9
    breakdown = metrics["energy_breakdown_by_node"][nid]
    assert breakdown["tx"] == pytest.approx(node.energy_tx)
    assert breakdown.get("ramp", 0.0) == pytest.approx(node.energy_ramp)
