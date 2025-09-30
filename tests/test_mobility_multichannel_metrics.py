from loraflexsim.launcher import Simulator, MultiChannel


def test_mobility_multichannel_metrics() -> None:
    params = dict(
        num_nodes=20,
        area_size=100,
        transmission_mode="Random",
        packet_interval=1.0,
        packets_to_send=10,
        mobility=True,
        seed=1,
        fixed_sf=7,
    )

    sim_single = Simulator(
        **params,
        channels=MultiChannel([868_100_000.0]),
    )
    sim_single.run()
    metrics_single = sim_single.get_metrics()

    sim_multi = Simulator(
        **params,
        channels=MultiChannel([868_100_000.0, 868_300_000.0, 868_500_000.0]),
    )
    sim_multi.run()
    metrics_multi = sim_multi.get_metrics()

    assert metrics_multi["PDR"] > metrics_single["PDR"]

    freqs = {
        e["frequency_hz"]
        for e in sim_multi._events_log_map.values()
        if e["result"] != "Mobility"
    }
    assert len(freqs) >= 2
