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

