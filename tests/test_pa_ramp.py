import random
from loraflexsim.launcher.channel import Channel


def test_pa_ramp_up_affects_rssi():
    random.seed(0)
    ch = Channel(phy_model="omnet", pa_ramp_up_s=1.0, shadowing_std=0.0, fast_fading_std=0.0)
    ch.omnet_phy.start_tx()
    ch.omnet_phy.update(0.5)
    r1, _ = ch.compute_rssi(14.0, 10.0)
    ch.omnet_phy.update(0.6)
    r2, _ = ch.compute_rssi(14.0, 10.0)
    assert r2 > r1
