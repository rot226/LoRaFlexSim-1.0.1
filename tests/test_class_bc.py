import math

from loraflexsim.launcher.downlink_scheduler import DownlinkScheduler
from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.lorawan import DR_TO_SF, next_beacon_time
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.server import NetworkServer


def test_schedule_beacon_time():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    t = scheduler.schedule_beacon(62.0, b"x", gw, beacon_interval=10.0)
    assert math.isclose(t, 70.0)
    entry = scheduler.pop_ready(0, 70.0)
    assert entry and entry.frame == b"x" and entry.gateway is gw


def test_schedule_class_b():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14, class_type="B")
    t = scheduler.schedule_class_b(
        node,
        0.0,
        b"a",
        gw,
        beacon_interval=128.0,
        ping_slot_interval=1.0,
        ping_slot_offset=2.0,
    )
    assert math.isclose(t, 2.0)
    entry = scheduler.pop_ready(node.id, t)
    assert entry and entry.frame == b"a" and entry.gateway is gw


def test_schedule_class_c_delay():
    scheduler = DownlinkScheduler(link_delay=0.5)
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14, class_type="C")
    t = scheduler.schedule_class_c(node, 1.0, b"b", gw)
    assert math.isclose(t, 1.5)
    entry = scheduler.pop_ready(node.id, t)
    assert entry and entry.frame == b"b" and entry.gateway is gw


def test_next_beacon_time_with_drift():
    interval = 128.0
    drift = 20e-6
    first = next_beacon_time(interval - 1e-3, interval, last_beacon=0.0, drift=drift)
    assert math.isclose(first, interval * (1.0 + drift), rel_tol=1e-9)
    second = next_beacon_time(first - 1e-3, interval, last_beacon=first, drift=drift)
    assert math.isclose(second, first + interval * (1.0 + drift), rel_tol=1e-9)


def test_next_beacon_time_after_successive_losses():
    interval = 128.0
    drift = 50e-6
    after = interval * 3.4
    expected = math.ceil((after + 1e-9) / interval) * interval
    t = next_beacon_time(after, interval, last_beacon=0.0, drift=drift)
    assert math.isclose(t, expected, rel_tol=1e-9)


def test_node_clock_offset_after_missed_beacons():
    interval = 128.0
    drift = 30e-6
    node = Node(1, 0.0, 0.0, 7, 14, class_type="B", beacon_drift=drift)
    node.clock_offset = 0.0
    node.miss_beacon(interval)
    assert math.isclose(node.clock_offset, interval * drift, rel_tol=1e-9)
    node.miss_beacon(interval)
    assert math.isclose(node.clock_offset, 2 * interval * drift, rel_tol=1e-9)
    node.miss_beacon(interval)
    # Drift is capped after two consecutive losses
    assert math.isclose(node.clock_offset, 2 * interval * drift, rel_tol=1e-9)


def test_class_b_explicit_rate_and_gateway_congestion():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14, class_type="B")
    node.ping_slot_periodicity = 0
    node.last_beacon_time = 0.0
    frame_a = b"a"
    frame_b = b"bb"
    t1 = scheduler.schedule_class_b(
        node,
        0.0,
        frame_a,
        gw,
        beacon_interval=128.0,
        ping_slot_interval=1.0,
        ping_slot_offset=2.0,
        data_rate=5,
        tx_power=10.0,
    )
    duration_first = node.channel.airtime(DR_TO_SF[5], len(frame_a))
    periodicity = max(0, min(7, node.ping_slot_periodicity or 0))
    slot_spacing = 1.0 * (2 ** (7 - periodicity))
    t2 = scheduler.schedule_class_b(
        node,
        0.0,
        frame_b,
        gw,
        beacon_interval=128.0,
        ping_slot_interval=1.0,
        ping_slot_offset=2.0,
        data_rate=4,
    )
    assert math.isclose(t1, 2.0)
    assert math.isclose(t2, t1 + slot_spacing)
    entry1 = scheduler.pop_ready(node.id, t1)
    assert entry1 and entry1.data_rate == 5 and entry1.tx_power == 10.0
    entry2 = scheduler.pop_ready(node.id, t2)
    assert entry2 and entry2.data_rate == 4 and entry2.tx_power is None


def test_class_b_priority_preemption():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14, class_type="B")
    node.ping_slot_periodicity = 0
    node.last_beacon_time = 0.0
    low_frame = b"low"
    high_frame = b"high"
    t_low = scheduler.schedule_class_b(
        node,
        0.0,
        low_frame,
        gw,
        beacon_interval=128.0,
        ping_slot_interval=1.0,
        ping_slot_offset=2.0,
        priority=0,
    )
    assert math.isclose(t_low, 2.0)
    t_high = scheduler.schedule_class_b(
        node,
        0.0,
        high_frame,
        gw,
        beacon_interval=128.0,
        ping_slot_interval=1.0,
        ping_slot_offset=2.0,
        priority=-1,
    )
    assert math.isclose(t_high, 2.0)
    first = scheduler.pop_ready(node.id, t_high)
    assert first and first.frame == high_frame
    periodicity = max(0, min(7, node.ping_slot_periodicity or 0))
    slot_spacing = 1.0 * (2 ** (7 - periodicity))
    second = scheduler.pop_ready(node.id, t_high + slot_spacing)
    assert second and second.frame == low_frame


def test_class_c_multi_gateway_latency_and_metadata():
    scheduler = DownlinkScheduler(link_delay=0.2)
    gw1 = Gateway(1, 0, 0)
    gw2 = Gateway(2, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14, class_type="C")
    frame1 = b"ping"
    frame2 = b"pong"
    t1 = scheduler.schedule_class_c(node, 1.0, frame1, gw1, data_rate=4, tx_power=6.0)
    t2 = scheduler.schedule_class_c(node, 1.0, frame2, gw2, data_rate=3, tx_power=12.0)
    assert math.isclose(t1, 1.2)
    assert math.isclose(t2, 1.2)
    entry1 = scheduler.pop_ready(node.id, t1)
    assert entry1 and entry1.gateway is gw1 and entry1.data_rate == 4 and entry1.tx_power == 6.0
    gw1.buffer_downlink(node.id, entry1.frame, data_rate=entry1.data_rate, tx_power=entry1.tx_power)
    stored1 = gw1.pop_downlink(node.id)
    assert stored1 == (frame1, 4, 6.0)
    next_time = scheduler.next_time(node.id)
    entry2 = scheduler.pop_ready(node.id, next_time)
    assert entry2 and entry2.gateway is gw2 and entry2.data_rate == 3 and entry2.tx_power == 12.0
    gw2.buffer_downlink(node.id, entry2.frame, data_rate=entry2.data_rate, tx_power=entry2.tx_power)
    stored2 = gw2.pop_downlink(node.id)
    assert stored2 == (frame2, 3, 12.0)


def test_network_server_class_b_uses_node_clock_offset():
    server = NetworkServer()
    gateway = Gateway(1, 0, 0)
    server.gateways = [gateway]
    server.beacon_interval = 128.0
    server.ping_slot_interval = 1.0
    server.ping_slot_offset = 2.0

    node = Node(1, 0.0, 0.0, 7, 14, class_type="B", beacon_drift=30e-6)
    node.ping_slot_periodicity = 0
    node.register_beacon(0.0)
    node.miss_beacon(server.beacon_interval)
    assert node.clock_offset != 0.0

    after = server.simulator.current_time if server.simulator else 0.0
    expected_slot = node.next_ping_slot_time(
        after,
        server.beacon_interval,
        server.ping_slot_interval,
        server.ping_slot_offset,
    )

    payload = b"offset-test"
    server.send_downlink(node, payload)
    scheduled_time = server.scheduler.next_time(node.id)

    assert math.isclose(scheduled_time, expected_slot, rel_tol=1e-9)
    entry = server.scheduler.pop_ready(node.id, scheduled_time)
    assert entry and entry.gateway is gateway
    delivered = entry.frame.payload if hasattr(entry.frame, "payload") else entry.frame
    assert delivered == payload
