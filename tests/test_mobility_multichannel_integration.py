from loraflexsim.launcher import Simulator, MultiChannel


def test_mobility_multichannel_integration() -> None:
    sim = Simulator(
        num_nodes=5,
        mobility=True,
        packets_to_send=1,
        channels=MultiChannel([868_100_000.0, 868_300_000.0]),
        seed=1,
    )
    sim.run()
    metrics = sim.get_metrics()
    assert metrics["PDR"] > 0
    freqs = {
        ev["frequency_hz"]
        for ev in sim._events_log_map.values()
        if ev["result"] != "Mobility"
    }
    assert len(freqs) >= 2
