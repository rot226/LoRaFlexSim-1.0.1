from loraflexsim.phy import LoRaPHY
from loraflexsim.loranode import Node


class DummyChannel:
    def airtime(self, sf, payload_size):
        return 0.0

    def compute_rssi(self, tx_power, distance, sf, **kwargs):
        return 0.0, 0.0

    def packet_error_rate(self, snr, sf, payload_bytes=20):
        return 0.0


def test_transmit_success_when_per_zero_and_rng_zero():
    ch = DummyChannel()
    n1 = Node(0, 0.0, 0.0, 7, 14, channel=ch)
    n2 = Node(1, 0.0, 0.0, 7, 14, channel=ch)
    phy = LoRaPHY(n1)

    class ZeroRng:
        def random(self):
            return 0.0

    rssi, snr, airtime, success = phy.transmit(n2, 10, rng=ZeroRng())
    assert success is True
