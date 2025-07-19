import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.server import NetworkServer  # noqa: E402
from VERSION_4.launcher.downlink_scheduler import DownlinkScheduler  # noqa: E402


def test_scheduled_downlink_delivery():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    gw = Gateway(1, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]
    ns.send_downlink(node, b"data", at_time=5.0)

    ns.deliver_scheduled(node.id, 4.0)
    assert gw.pop_downlink(node.id) is None

    ns.deliver_scheduled(node.id, 5.0)
    frame = gw.pop_downlink(node.id)
    assert frame is not None
    assert isinstance(frame.payload, bytes) and frame.payload == b"data"


def test_multiple_scheduled_frames():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    gw = Gateway(1, 0.0, 0.0)
    ns = NetworkServer()
    ns.gateways = [gw]
    ns.nodes = [node]
    ns.send_downlink(node, b"one", at_time=1.0)
    ns.send_downlink(node, b"two", at_time=1.0)

    ns.deliver_scheduled(node.id, 1.0)
    frames = [gw.pop_downlink(node.id), gw.pop_downlink(node.id)]
    payloads = sorted(f.payload for f in frames if f is not None)
    assert payloads == [b"one", b"two"]
    assert gw.pop_downlink(node.id) is None


def test_scheduler_conflict_resolution():
    node1 = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    node1.class_type = "B"
    node2 = Node(2, 0.0, 0.0, 7, 14.0, channel=Channel())
    node2.class_type = "B"
    gw = Gateway(1, 0.0, 0.0)
    sched = DownlinkScheduler()
    from VERSION_4.launcher.lorawan import LoRaWANFrame

    frame = LoRaWANFrame(mhdr=0x60, fctrl=0, fcnt=0, payload=b"a")
    t1 = sched.schedule_class_b(
        node1,
        0.0,
        frame,
        gw,
        128.0,
        1.0,
        2.0,
    )
    t2 = sched.schedule_class_b(
        node2,
        0.0,
        frame,
        gw,
        128.0,
        1.0,
        2.0,
    )
    assert t2 > t1


def test_schedule_class_c_returns_time():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    node.class_type = "C"
    gw = Gateway(1, 0.0, 0.0)
    sched = DownlinkScheduler()
    from VERSION_4.launcher.lorawan import LoRaWANFrame

    frame = LoRaWANFrame(mhdr=0x60, fctrl=0, fcnt=0, payload=b"a")
    t0 = sched.schedule_class_c(node, 0.0, frame, gw)
    assert t0 == 0.0
    # Scheduling another frame before gateway is free should defer it
    t1 = sched.schedule_class_c(node, 0.0, frame, gw)
    assert t1 > t0
