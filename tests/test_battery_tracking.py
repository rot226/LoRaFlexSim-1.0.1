import pytest

from loraflexsim.launcher.simulator import Simulator


def test_battery_tracking():
    sim = Simulator(num_nodes=3, packets_to_send=2, battery_capacity_j=1000, seed=1)

    previous = {n.id: n.battery_remaining_j for n in sim.nodes}
    prev_time = sim.current_time

    while sim.event_queue:
        sim.run(max_steps=1)
        for n in sim.nodes:
            n.consume_until(sim.current_time)
        for n in sim.nodes:
            assert n.battery_remaining_j <= previous[n.id] + 1e-9
        if sim.current_time - prev_time > 1e-9:
            assert any(
                n.battery_remaining_j < previous[n.id] - 1e-9 for n in sim.nodes
            )
        previous = {n.id: n.battery_remaining_j for n in sim.nodes}
        prev_time = sim.current_time

    for n in sim.nodes:
        assert n.battery_remaining_j < 1000
