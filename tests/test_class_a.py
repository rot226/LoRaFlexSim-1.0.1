import math
from types import SimpleNamespace

from loraflexsim.launcher.downlink_scheduler import DownlinkScheduler
from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.server import NetworkServer
from loraflexsim.launcher.lorawan import compute_rx2


def test_schedule_class_a():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14)
    t = scheduler.schedule_class_a(node, 0.0, 1.0, 2.0, b"a", gw)
    assert math.isclose(t, 1.0)
    entry = scheduler.pop_ready(node.id, t)
    assert entry and entry.frame == b"a" and entry.gateway is gw


def test_schedule_class_a_after_delay():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14)
    scheduler._gateway_busy[gw.id] = 1.5
    t = scheduler.schedule_class_a(node, 0.0, 1.0, 2.0, b"b", gw)
    assert math.isclose(t, 2.0)
    entry = scheduler.pop_ready(node.id, t)
    assert entry and entry.frame == b"b" and entry.gateway is gw


def test_schedule_class_a_rejects_after_rx2():
    scheduler = DownlinkScheduler()
    gw = Gateway(1, 0, 0)
    node = Node(1, 0.0, 0.0, 7, 14)
    scheduler._gateway_busy[gw.id] = 3.0
    t = scheduler.schedule_class_a(node, 0.0, 1.0, 2.0, b"c", gw)
    assert t is None
    assert scheduler.pop_ready(node.id, 10.0) is None


def test_downlink_with_server_delays():
    sim = SimpleNamespace(current_time=0.0, event_queue=[], event_id_counter=0)
    server = NetworkServer(simulator=sim, process_delay=1.0, network_delay=1.0)
    gw = Gateway(0, 0, 0)
    server.gateways = [gw]
    node = Node(0, 0, 0, 7, 14)
    server.nodes = [node]

    server.schedule_receive(1, node.id, gw.id, -40, at_time=0.0)
    evt = sim.event_queue.pop(0)
    sim.current_time = evt.time
    server._handle_network_arrival(evt.id)
    evt = sim.event_queue.pop(0)
    sim.current_time = evt.time
    server._process_scheduled(evt.id)

    server.send_downlink(node, b"x")
    t = server.scheduler.next_time(node.id)
    expected = compute_rx2(0.0, node.rx_delay) + server.network_delay
    assert math.isclose(t, expected)

