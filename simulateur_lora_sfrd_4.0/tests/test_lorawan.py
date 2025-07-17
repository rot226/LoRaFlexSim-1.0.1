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
    ADRParamSetupAns,
    ForceRejoinReq,
    RejoinParamSetupReq,
    RejoinParamSetupAns,
    RekeyInd,
    RekeyConf,
    DeviceModeInd,
    DeviceModeConf,
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


def test_adr_param_setup_req_roundtrip():
    req = ADRParamSetupReq(5, 9)
    data = req.to_bytes()
    parsed = ADRParamSetupReq.from_bytes(data)
    assert parsed == req


def test_adr_param_setup_ans_roundtrip():
    ans = ADRParamSetupAns()
    data = ans.to_bytes()
    parsed = ADRParamSetupAns.from_bytes(data)
    assert isinstance(parsed, ADRParamSetupAns)


def test_force_rejoin_req_roundtrip():
    req = ForceRejoinReq(2, 10, 5)
    data = req.to_bytes()
    parsed = ForceRejoinReq.from_bytes(data)
    assert parsed == req


def test_rejoin_param_setup_roundtrip():
    req = RejoinParamSetupReq(3, 7)
    data = req.to_bytes()
    parsed = RejoinParamSetupReq.from_bytes(data)
    assert parsed == req
    ans = RejoinParamSetupAns()
    data_ans = ans.to_bytes()
    parsed_ans = RejoinParamSetupAns.from_bytes(data_ans)
    assert parsed_ans == ans


def test_rekey_roundtrip():
    ind = RekeyInd(1)
    data = ind.to_bytes()
    parsed_ind = RekeyInd.from_bytes(data)
    assert parsed_ind == ind
    conf = RekeyConf(1)
    data_conf = conf.to_bytes()
    parsed_conf = RekeyConf.from_bytes(data_conf)
    assert parsed_conf == conf


def test_device_mode_roundtrip():
    ind = DeviceModeInd("B")
    data = ind.to_bytes()
    parsed_ind = DeviceModeInd.from_bytes(data)
    assert parsed_ind == ind
    conf = DeviceModeConf("B")
    data_conf = conf.to_bytes()
    parsed_conf = DeviceModeConf.from_bytes(data_conf)
    assert parsed_conf == conf


def test_rx_delay_affects_receive_windows():
    from VERSION_4.launcher.node import Node
    from VERSION_4.launcher.channel import Channel

    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    node.rx_delay = 3
    rx1, rx2 = node.schedule_receive_windows(10.0)
    assert rx1 == pytest.approx(13.0)
    assert rx2 == pytest.approx(14.0)

