import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.lorawan import (  # noqa: E402
    LoRaWANFrame,
    DutyCycleReq,
    RXParamSetupReq,
    DevStatusReq,
    DevStatusAns,
    PingSlotChannelReq,
    PingSlotInfoReq,
    PingSlotInfoAns,
    BeaconTimingReq,
    BeaconTimingAns,
    ResetConf,
    ResetInd,
    ADRParamSetupReq,
    ADRParamSetupAns,
    RekeyInd,
    RekeyConf,
    RejoinParamSetupReq,
    RejoinParamSetupAns,
    ForceRejoinReq,
    DeviceModeInd,
    DeviceModeConf,
)


def test_handle_duty_cycle_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, DutyCycleReq(3).to_bytes())
    node.handle_downlink(frame)
    assert node.max_duty_cycle == 3


def test_handle_rx_param_setup_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    req = RXParamSetupReq(1, 5, 869525000)
    frame = LoRaWANFrame(0, 0, 0, req.to_bytes())
    node.handle_downlink(frame)
    assert node.rx1_dr_offset == 1
    assert node.rx2_datarate == 5
    assert node.rx2_frequency == 869525000


def test_handle_dev_status_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, DevStatusReq().to_bytes())
    node.handle_downlink(frame)
    ans = DevStatusAns.from_bytes(node.pending_mac_cmd)
    assert ans.battery == 255


def test_handle_ping_slot_channel_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    req = PingSlotChannelReq(869525000, 3)
    frame = LoRaWANFrame(0, 0, 0, req.to_bytes())
    node.handle_downlink(frame)
    assert node.ping_slot_frequency == 869525000
    assert node.ping_slot_dr == 3


def test_handle_ping_slot_info_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    req = PingSlotInfoReq(2)
    frame = LoRaWANFrame(0, 0, 0, req.to_bytes())
    node.handle_downlink(frame)
    assert node.ping_slot_periodicity == 2
    assert node.pending_mac_cmd == PingSlotInfoAns().to_bytes()


def test_handle_beacon_timing_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, BeaconTimingReq().to_bytes())
    node.handle_downlink(frame)
    ans = BeaconTimingAns.from_bytes(node.pending_mac_cmd)
    assert ans.delay == 0
    assert ans.channel == 0


def test_handle_reset_conf():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, ResetConf(1).to_bytes())
    node.handle_downlink(frame)
    assert node.lorawan_minor == 1
    assert node.pending_mac_cmd == ResetInd(1).to_bytes()


def test_handle_adr_param_setup_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, ADRParamSetupReq(2, 4).to_bytes())
    node.handle_downlink(frame)
    assert node.adr_ack_limit == 2
    assert node.adr_ack_delay == 4
    assert node.pending_mac_cmd == ADRParamSetupAns().to_bytes()


def test_handle_rekey_ind():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, RekeyInd(1).to_bytes())
    node.handle_downlink(frame)
    assert node.rekey_key_type == 1
    assert node.pending_mac_cmd == RekeyConf(1).to_bytes()


def test_handle_rejoin_param_setup_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, RejoinParamSetupReq(3, 5).to_bytes())
    node.handle_downlink(frame)
    assert node.rejoin_time_n == 3
    assert node.rejoin_count_n == 5
    assert node.pending_mac_cmd == RejoinParamSetupAns().to_bytes()


def test_handle_force_rejoin_req():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, ForceRejoinReq(10, 2).to_bytes())
    node.handle_downlink(frame)
    assert node.force_rejoin_period == 10
    assert node.force_rejoin_type == 2


def test_handle_device_mode_ind():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel())
    frame = LoRaWANFrame(0, 0, 0, DeviceModeInd("C").to_bytes())
    node.handle_downlink(frame)
    assert node.class_type == "C"
    assert node.pending_mac_cmd == DeviceModeConf("C").to_bytes()

