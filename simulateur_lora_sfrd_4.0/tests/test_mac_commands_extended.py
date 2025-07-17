import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node
from VERSION_4.launcher.channel import Channel
from VERSION_4.launcher.lorawan import (
    LoRaWANFrame,
    DutyCycleReq,
    RXParamSetupReq,
    DevStatusReq,
    DevStatusAns,
    PingSlotChannelReq,
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
