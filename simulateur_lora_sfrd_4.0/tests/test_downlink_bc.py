import sys
from pathlib import Path
import heapq

# Allow importing the VERSION_4 package from the repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.simulator import Simulator, Event, EventType  # noqa: E402


def _make_sim(class_type: str) -> Simulator:
    ch = Channel(shadowing_std=0)
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=10.0,
        transmission_mode="Periodic",
        packet_interval=10.0,
        packets_to_send=1,
        mobility=False,
        duty_cycle=None,
        channels=[ch],
        fixed_sf=7,
        fixed_tx_power=14.0,
    )
    node = sim.nodes[0]
    node.class_type = class_type
    node.x = sim.gateways[0].x
    node.y = sim.gateways[0].y
    return sim


def test_downlink_delivery_class_c():
    sim = _make_sim("C")
    node = sim.nodes[0]
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    sim.network_server.send_downlink(node, b"data")
    while sim.step():
        if sim.current_time > sim.class_c_rx_interval + 0.1:
            break
    assert node.fcnt_down == 1
    assert node.downlink_pending == 0


def test_scheduler_programs_class_b_slot():
    sim = _make_sim("B")
    node = sim.nodes[0]
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    heapq.heappush(sim.event_queue, Event(0.0, EventType.BEACON, 1, 0))
    from VERSION_4.launcher.lorawan import LoRaWANFrame

    frame = LoRaWANFrame(mhdr=0x60, fctrl=0, fcnt=0, payload=b"data")
    t = sim.network_server.scheduler.schedule_class_b(
        node,
        0.0,
        frame,
        sim.gateways[0],
        sim.beacon_interval,
        sim.ping_slot_interval,
        sim.ping_slot_offset,
    )
    while sim.step():
        if sim.current_time > t + 0.1:
            break
    assert node.fcnt_down == 1
    assert node.downlink_pending == 0
    assert t == sim.ping_slot_offset


def test_ping_slot_periodicity():
    sim = _make_sim("B")
    node = sim.nodes[0]
    node.ping_slot_periodicity = 1
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    heapq.heappush(sim.event_queue, Event(0.0, EventType.BEACON, 1, 0))
    from VERSION_4.launcher.lorawan import LoRaWANFrame

    frame = LoRaWANFrame(mhdr=0x60, fctrl=0, fcnt=0, payload=b"data")
    t = sim.network_server.scheduler.schedule_class_b(
        node,
        5.0,
        frame,
        sim.gateways[0],
        sim.beacon_interval,
        sim.ping_slot_interval,
        sim.ping_slot_offset,
    )
    while sim.step():
        if sim.current_time > t + 0.1:
            break
    assert node.fcnt_down == 1
    assert t == 6.0


def test_class_c_continuous_rx():
    sim = _make_sim("C")
    node = sim.nodes[0]
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    sim.network_server.send_downlink(node, b"data", at_time=1.5)
    while sim.step():
        if sim.current_time > 2.1:
            break
    assert node.fcnt_down == 1



def test_class_b_downlink_buffer_delivery():
    sim = _make_sim("B")
    node = sim.nodes[0]
    sim.event_queue.clear()
    sim.event_id_counter = 0
    heapq.heappush(sim.event_queue, Event(0.0, EventType.RX_WINDOW, 0, node.id))
    heapq.heappush(sim.event_queue, Event(0.0, EventType.BEACON, 1, 0))
    sim.network_server.send_downlink(node, b"data")
    limit = sim.ping_slot_offset + 0.1
    while sim.step():
        if sim.current_time > limit:
            break
    assert node.fcnt_down == 1
    assert node.downlink_pending == 0


def test_class_c_downlink_after_tx_window():
    sim = _make_sim("C")
    node = sim.nodes[0]
    sim.event_queue.clear()
    sim.event_id_counter = 0
    sim.schedule_event(node, 0.0)
    duration = node.channel.airtime(node.sf, payload_size=sim.payload_size_bytes)
    rx1, _ = node.schedule_receive_windows(duration)
    sim.network_server.send_downlink(node, b"data", at_time=rx1)
    while sim.step():
        if sim.current_time > rx1 + 0.1:
            break
    assert node.fcnt_down == 1
    assert node.downlink_pending == 0
