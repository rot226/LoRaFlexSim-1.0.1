import math
import pytest

from loraflexsim.launcher.channel import Channel
try:
    from loraflexsim.launcher.flora_cpp import FloraCppPHY
except OSError:
    FloraCppPHY = None


@pytest.mark.parametrize("distance", [100.0, 1000.0])
@pytest.mark.parametrize("sf", [7, 12])
@pytest.mark.parametrize("bandwidth", [125000, 500000])
def test_flora_equivalence(distance, sf, bandwidth):
    """FLoRa C++ library and Python model should match for path loss, RSSI and PER."""
    ch_cpp = Channel(
        phy_model="flora_cpp",
        bandwidth=bandwidth,
        shadowing_std=0.0,
        use_flora_curves=True,
    )
    if FloraCppPHY is None or not isinstance(ch_cpp.flora_phy, FloraCppPHY):
        pytest.skip("libflora_phy.so missing")
    ch_py = Channel(
        phy_model="flora_full",
        bandwidth=bandwidth,
        shadowing_std=0.0,
        use_flora_curves=True,
    )
    tx_power = 14.0
    loss_cpp = ch_cpp.path_loss(distance)
    loss_py = ch_py.path_loss(distance)
    assert loss_py == pytest.approx(loss_cpp, abs=0.01)
    rssi_cpp, _ = ch_cpp.compute_rssi(tx_power, distance, sf=sf)
    rssi_py, _ = ch_py.compute_rssi(tx_power, distance, sf=sf)
    assert rssi_py == pytest.approx(rssi_cpp, abs=0.01)
    noise_cpp = ch_cpp._flora_noise_dBm(sf)
    noise_py = ch_py._flora_noise_dBm(sf)
    snr_cpp = rssi_cpp - noise_cpp + ch_cpp.snr_offset_dB
    snr_py = rssi_py - noise_py + ch_py.snr_offset_dB
    assert snr_py == pytest.approx(snr_cpp, abs=0.01)
    per_cpp = ch_cpp.packet_error_rate(snr_cpp, sf)
    per_py = ch_py.packet_error_rate(snr_py, sf)
    assert per_py == pytest.approx(per_cpp, abs=0.01)
