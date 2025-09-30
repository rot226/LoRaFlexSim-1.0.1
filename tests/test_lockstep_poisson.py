from loraflexsim.launcher.simulator import Simulator


def test_lockstep_poisson_sequences():
    params = dict(
        num_nodes=2,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=5.0,
        packets_to_send=5,
        duty_cycle=0.01,
        mobility=False,
        seed=123,
        lock_step_poisson=True,
    )
    sim1 = Simulator(**params)
    sim1.run()
    seqs1 = [tuple(n.precomputed_arrivals) for n in sim1.nodes]

    sim2 = Simulator(**params)
    sim2.run()
    seqs2 = [tuple(n.precomputed_arrivals) for n in sim2.nodes]

    assert seqs1 == seqs2

    sim1.get_metrics()
    sim2.get_metrics()

    assert seqs1 == [tuple(n.precomputed_arrivals) for n in sim1.nodes]
    assert seqs2 == [tuple(n.precomputed_arrivals) for n in sim2.nodes]

