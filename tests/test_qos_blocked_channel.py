from loraflexsim.launcher.simulator import Simulator


def test_blocked_channel_events_are_skipped():
    simulator = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=1.0,
        first_packet_interval=0.1,
        mobility=False,
        warm_up_intervals=0,
        packets_to_send=0,
        seed=1,
    )

    node = simulator.nodes[0]
    simulator.event_queue.clear()
    node._qos_blocked_channel = True
    node.channel = None

    simulator.schedule_event(node, simulator.current_time, reason="blocked-test")

    assert not simulator.event_queue
    assert node.out_of_service
    assert simulator.out_of_service_queue
    reason = simulator.out_of_service_queue[-1][1]
    assert reason in {"blocked_channel", "missing_channel"}
    assert simulator.step() is False

