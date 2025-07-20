import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from VERSION_4.launcher.lorawan import (  # noqa: E402
    NewChannelReq,
    RXParamSetupReq,
    DevStatusAns,
    PingSlotInfoReq,
    BeaconTimingAns,
    ADRParamSetupReq,
    RejoinParamSetupReq,
    DeviceModeInd,
)


def test_new_channel_req_roundtrip():
    req = NewChannelReq(1, 868300000, 0x22)
    data = req.to_bytes()
    parsed = NewChannelReq.from_bytes(data)
    assert parsed == req


def test_rx_param_setup_req_roundtrip():
    req = RXParamSetupReq(3, 5, 869525000)
    data = req.to_bytes()
    parsed = RXParamSetupReq.from_bytes(data)
    assert parsed == req


def test_dev_status_ans_roundtrip():
    ans = DevStatusAns(battery=200, margin=10)
    data = ans.to_bytes()
    parsed = DevStatusAns.from_bytes(data)
    assert parsed == ans


def test_ping_slot_info_req_roundtrip():
    req = PingSlotInfoReq(5)
    data = req.to_bytes()
    parsed = PingSlotInfoReq.from_bytes(data)
    assert parsed == req


def test_beacon_timing_ans_roundtrip():
    ans = BeaconTimingAns(256, 3)
    data = ans.to_bytes()
    parsed = BeaconTimingAns.from_bytes(data)
    assert parsed == ans


def test_rx_delay_affects_receive_windows():
    from VERSION_4.launcher.node import Node
    from VERSION_4.launcher.channel import Channel

    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    node.rx_delay = 3
    rx1, rx2 = node.schedule_receive_windows(10.0)
    assert rx1 == pytest.approx(13.0)
    assert rx2 == pytest.approx(14.0)


def test_next_ping_slot_time_from_last_beacon():
    from VERSION_4.launcher.node import Node
    from VERSION_4.launcher.channel import Channel

    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    node.class_type = "B"
    node.last_beacon_time = 100.0
    node.ping_slot_periodicity = 1

    t1 = node.next_ping_slot_time(
        current_time=100.0,
        beacon_interval=120.0,
        ping_slot_interval=2.0,
        ping_slot_offset=0.5,
    )
    assert t1 == pytest.approx(100.5)

    t2 = node.next_ping_slot_time(
        current_time=102.5,
        beacon_interval=120.0,
        ping_slot_interval=2.0,
        ping_slot_offset=0.5,
    )
    assert t2 == pytest.approx(104.5)


def test_next_ping_slot_time_drift():
    from VERSION_4.launcher.node import Node
    from VERSION_4.launcher.channel import Channel

    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), beacon_drift=0.001)
    node.class_type = "B"
    node.last_beacon_time = 0.0

    t = node.next_ping_slot_time(
        current_time=0.0,
        beacon_interval=10.0,
        ping_slot_interval=1.0,
        ping_slot_offset=0.5,
    )
    assert t == pytest.approx(0.501)


def test_adr_param_setup_req_roundtrip():
    req = ADRParamSetupReq(3, 5)
    data = req.to_bytes()
    parsed = ADRParamSetupReq.from_bytes(data)
    assert parsed == req


def test_rejoin_param_setup_req_roundtrip():
    req = RejoinParamSetupReq(2, 7)
    data = req.to_bytes()
    parsed = RejoinParamSetupReq.from_bytes(data)
    assert parsed == req


def test_device_mode_ind_roundtrip():
    ind = DeviceModeInd("C")
    data = ind.to_bytes()
    parsed = DeviceModeInd.from_bytes(data)
    assert parsed == ind


def test_ping_slot_channel_ans_roundtrip():
    from VERSION_4.launcher.lorawan import PingSlotChannelAns

    ans = PingSlotChannelAns(status=3)
    data = ans.to_bytes()
    parsed = PingSlotChannelAns.from_bytes(data)
    assert parsed == ans


def test_beacon_freq_ans_roundtrip():
    from VERSION_4.launcher.lorawan import BeaconFreqAns

    ans = BeaconFreqAns(status=1)
    data = ans.to_bytes()
    parsed = BeaconFreqAns.from_bytes(data)
    assert parsed == ans


def test_next_beacon_time_drift():
    from VERSION_4.launcher.lorawan import next_beacon_time

    t = next_beacon_time(0.1, 10.0, last_beacon=0.0, drift=0.1)
    assert t == pytest.approx(11.0)


def test_next_beacon_time_recover():
    from VERSION_4.launcher.lorawan import next_beacon_time

    t = next_beacon_time(35.0, 10.0, last_beacon=0.0, drift=0.0, loss_limit=2.0)
    assert t == pytest.approx(40.0)


def test_join_server_invalid_key_and_rejoin():
    from VERSION_4.launcher.lorawan import (
        JoinServer,
        JoinRequest,
        JoinAccept,
        compute_join_mic,
        aes_encrypt,
    )

    js = JoinServer(net_id=1)
    app_key = bytes(range(16))
    js.register(1, 2, app_key)

    req = JoinRequest(1, 2, 1)
    req.mic = compute_join_mic(app_key, req.to_bytes())
    accept, nwk, app = js.handle_join(req)
    assert isinstance(accept, JoinAccept)
    assert len(nwk) == 16
    assert len(app) == 16
    assert accept.mic == compute_join_mic(app_key, accept.to_bytes())
    assert aes_encrypt(app_key, accept.encrypted)[:10] == accept.to_bytes()

    with pytest.raises(ValueError):
        js.handle_join(req)

    bad = JoinRequest(1, 3, 1)
    bad.mic = compute_join_mic(app_key, bad.to_bytes())
    with pytest.raises(KeyError):
        js.handle_join(bad)

    req2 = JoinRequest(1, 2, 2)
    req2.mic = compute_join_mic(app_key, req2.to_bytes())
    js.handle_join(req2)

