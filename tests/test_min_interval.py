from loraflexsim.launcher.simulator import Simulator
from loraflexsim.launcher.node import Node
import numpy_stub as np


def test_next_event_after_airtime():
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        transmission_mode="Random",
        packet_interval=1.0,
        packets_to_send=5,
        pure_poisson_mode=True,
        mobility=False,
        seed=1,
    )
    sim.run()
    events = sorted(sim._events_log_map.values(), key=lambda e: e["start_time"])
    for i in range(len(events) - 1):
        airtime = events[i]["end_time"] - events[i]["start_time"]
        delta = events[i + 1]["start_time"] - events[i]["start_time"]
        assert delta >= airtime


def test_poisson_min_interval_resampling():
    rng = np.random.Generator(np.random.MT19937(0))
    node = Node(0, 0, 0, 7, 20)
    node.ensure_poisson_arrivals(10.0, 0.1, rng, min_interval=1.0, limit=10)
    last = 0.0
    for t in node.arrival_queue:
        assert t - last >= 1.0
        last = t
