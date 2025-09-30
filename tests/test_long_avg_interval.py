from loraflexsim.launcher.simulator import Simulator


def test_long_term_avg_interval():
    mean_interval = 5.0
    count = 10_000
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=mean_interval,
        packets_to_send=count,
        duty_cycle=0.01,
        pure_poisson_mode=True,
        mobility=False,
        seed=0,
    )
    sim.run()
    node = sim.nodes[0]
    total_time = node._last_arrival_time
    average = total_time / node.packets_sent
    assert node.packets_sent == count
    assert abs(average - mean_interval) / mean_interval < 0.01
