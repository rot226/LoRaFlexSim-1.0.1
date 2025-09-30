from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.lorawan import (
    REGION_DEFAULT_RX2_DR,
    default_downlink_datarate,
)
from loraflexsim.launcher.node import Node


def test_default_downlink_datarate_mapping():
    assert default_downlink_datarate(None) is None
    for region, expected in REGION_DEFAULT_RX2_DR.items():
        channel = Channel(region=region)
        node = Node(1, 0.0, 0.0, 7, 14, channel=channel)
        assert default_downlink_datarate(region) == expected
        assert node.rx2_datarate == expected
        assert node.ping_slot_dr == expected
