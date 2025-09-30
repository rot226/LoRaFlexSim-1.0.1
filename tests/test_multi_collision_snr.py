import pytest

from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.channel import Channel


def test_multi_collision_snr_penalty():
    """SNR should degrade with the sum of simultaneous powers."""

    ch = Channel(shadowing_std=0.0, fast_fading_std=0.0, time_variation_std=0.0)
    sim = Simulator(
        num_nodes=3,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1000.0,
        packets_to_send=1,
        mobility=False,
        fixed_sf=7,
        fixed_tx_power=-50.0,
        pure_poisson_mode=True,
        channels=[ch],
    )

    nodes = sim.nodes
    for n in nodes:
        sim.schedule_event(n, 0.0)

    sim.step()  # node 0
    snr0 = nodes[0].last_snr
    sim.step()  # node 1, should see interference from node 0
    snr1 = nodes[1].last_snr
    sim.step()  # node 2, interfered by node 0 and 1
    snr2 = nodes[2].last_snr

    while sim.step():
        pass

    assert snr1 < snr0
    assert snr2 < snr1
    assert snr1 - snr2 == pytest.approx(3.0, abs=1.0)

