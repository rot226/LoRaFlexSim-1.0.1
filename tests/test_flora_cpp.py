import math
import pytest

from loraflexsim.launcher.channel import Channel


def test_rssi_snr_match_python_impl():
    try:
        ch_cpp = Channel(phy_model="flora_cpp", shadowing_std=0.0, use_flora_curves=True)
    except OSError:
        pytest.skip("libflora_phy.so missing")

    ch_py = Channel(phy_model="flora_full", shadowing_std=0.0, use_flora_curves=True)

    rssi_cpp, _ = ch_cpp.compute_rssi(14.0, 100.0, sf=7)
    rssi_py, snr_py = ch_py.compute_rssi(14.0, 100.0, sf=7)
    # Derive SNR using the FLoRa noise table to mimic the C++ implementation
    noise_cpp = ch_cpp._flora_noise_dBm(7)
    snr_cpp = rssi_cpp - noise_cpp + ch_cpp.snr_offset_dB

    assert abs(rssi_cpp - rssi_py) <= 0.01
    assert abs(snr_cpp - snr_py) <= 0.01
