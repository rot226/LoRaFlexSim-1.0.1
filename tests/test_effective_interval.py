from loraflexsim.launcher.simulator import Simulator


def test_effective_avg_interval():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=2.0,
        packets_to_send=50,
        duty_cycle=0.01,
        pure_poisson_mode=True,
        mobility=False,
        dump_intervals=False,
        seed=0,
    )
    sim.run()
    metrics = sim.get_metrics()
    assert abs(metrics["avg_arrival_interval_s"] - 2.0) / 2.0 < 0.1
