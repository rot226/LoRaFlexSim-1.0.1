import math
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.flora_phy import FloraPHY


def flora_equations(tx_power: float, distance: float, sf: int, ch: Channel):
    """Return RSSI and SNR using simplified FLoRa formulas."""
    loss = (
        FloraPHY.PATH_LOSS_D0
        + 10 * ch.path_loss_exp * math.log10(max(distance, 1.0) / FloraPHY.REFERENCE_DISTANCE)
    )
    rssi = (
        tx_power
        + ch.tx_antenna_gain_dB
        + ch.rx_antenna_gain_dB
        - loss
        - ch.cable_loss_dB
        + ch.rssi_offset_dB
    )
    noise = ch.FLORA_SENSITIVITY[sf][int(ch.bandwidth)]
    snr = rssi - noise + ch.snr_offset_dB
    if ch.processing_gain:
        snr += 10 * math.log10(2 ** sf)
    return rssi, snr


def oulu_equations(tx_power: float, distance: float, sf: int, ch: Channel):
    """Return RSSI and SNR using the Oulu path loss model."""
    loss = (
        FloraPHY.OULU_B
        + 10 * FloraPHY.OULU_N * math.log10(max(distance, 1.0) / FloraPHY.OULU_D0)
        - FloraPHY.OULU_ANTENNA_GAIN
    )
    rssi = (
        tx_power
        + ch.tx_antenna_gain_dB
        + ch.rx_antenna_gain_dB
        - loss
        - ch.cable_loss_dB
        + ch.rssi_offset_dB
    )
    noise = ch.FLORA_SENSITIVITY[sf][int(ch.bandwidth)]
    snr = rssi - noise + ch.snr_offset_dB
    if ch.processing_gain:
        snr += 10 * math.log10(2 ** sf)
    return rssi, snr


def hata_equations(tx_power: float, distance: float, sf: int, ch: Channel):
    """Return RSSI and SNR using the Hata-Okumura model."""
    loss = ch.hata_k1 + ch.hata_k2 * math.log10(max(distance, 1.0) / 1000.0)
    rssi = (
        tx_power
        + ch.tx_antenna_gain_dB
        + ch.rx_antenna_gain_dB
        - loss
        - ch.cable_loss_dB
        + ch.rssi_offset_dB
    )
    noise = ch.FLORA_SENSITIVITY[sf][int(ch.bandwidth)]
    snr = rssi - noise + ch.snr_offset_dB
    if ch.processing_gain:
        snr += 10 * math.log10(2 ** sf)
    return rssi, snr


def test_channel_compute_rssi_matches_flora_equations():
    tx_power = 14.0
    distance = 100.0
    sf = 7
    ch = Channel(environment="flora", phy_model="flora_full")
    ch.shadowing_std = 0.0  # deterministic
    expected_rssi, expected_snr = flora_equations(tx_power, distance, sf, ch)
    rssi, snr = ch.compute_rssi(tx_power, distance, sf=sf)
    assert abs(rssi - expected_rssi) <= 0.01
    assert abs(snr - expected_snr) <= 0.01


def test_oulu_path_loss_model():
    tx_power = 14.0
    distance = 500.0
    sf = 7
    ch = Channel(phy_model="flora_full", flora_loss_model="oulu")
    ch.shadowing_std = 0.0
    expected_rssi, expected_snr = oulu_equations(tx_power, distance, sf, ch)
    rssi, snr = ch.compute_rssi(tx_power, distance, sf=sf)
    assert abs(rssi - expected_rssi) <= 0.01
    assert abs(snr - expected_snr) <= 0.01


def test_hata_path_loss_model():
    tx_power = 14.0
    distance = 500.0
    sf = 7
    ch = Channel(phy_model="flora_full", flora_loss_model="hata")
    ch.shadowing_std = 0.0
    expected_rssi, expected_snr = hata_equations(tx_power, distance, sf, ch)
    rssi, snr = ch.compute_rssi(tx_power, distance, sf=sf)
    assert abs(rssi - expected_rssi) <= 0.01
    assert abs(snr - expected_snr) <= 0.01


def test_processing_gain_flag_adds_expected_offset():
    tx_power = 14.0
    distance = 100.0
    sf = 9
    ch = Channel(environment="flora", phy_model="flora_full", processing_gain=True)
    ch.shadowing_std = 0.0
    expected_rssi, expected_snr = flora_equations(tx_power, distance, sf, ch)
    rssi, snr = ch.compute_rssi(tx_power, distance, sf=sf)
    assert abs(rssi - expected_rssi) <= 0.01
    assert abs(snr - expected_snr) <= 0.01
