from loraflexsim.phy import LoRaPHY
from loraflexsim.loranode import Node
from loraflexsim.launcher.channel import Channel
from traffic.rng_manager import RngManager


def test_loraphy_transmit_deterministic():
    ch1 = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    ch2 = Channel(shadowing_std=0.0, fast_fading_std=0.0, noise_floor_std=0.0)
    n1 = Node(0, 0, 0, 7, 14, channel=ch1)
    n2 = Node(1, 0, 0, 7, 14, channel=ch2)

    phy = LoRaPHY(n1)
    mgr1 = RngManager(0)
    rng1 = mgr1.get_stream("phy", 0)
    res1 = phy.transmit(n2, 20, rng=rng1)
    mgr2 = RngManager(0)
    rng2 = mgr2.get_stream("phy", 0)
    res2 = phy.transmit(n2, 20, rng=rng2)
    assert res1 == res2
