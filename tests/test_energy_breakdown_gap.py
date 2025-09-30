import pytest

from loraflexsim.launcher.simulator import Simulator


def test_energy_breakdown_reports_pa_ramp_component():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        seed=0,
    )
    sim.run()
    node = sim.nodes[0]
    breakdown = node.get_energy_breakdown()
    assert "ramp" in breakdown
    assert breakdown["ramp"] == pytest.approx(node.energy_ramp)
