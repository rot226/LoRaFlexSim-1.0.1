from loraflexsim.launcher.simulator import Simulator, EventType


def test_first_packet_respects_delay():
    sim = Simulator(
        num_nodes=5,
        num_gateways=1,
        flora_mode=True,
        mobility=False,
        packets_to_send=1,
        seed=42,
    )
    start_times = [e.time for e in sim.event_queue if e.type == EventType.TX_START]
    assert all(t >= 5.0 for t in start_times)
