import math

import pytest

from loraflexsim.launcher.channel import Channel


def test_qos_path_gain_matches_formula():
    channel = Channel(frequency_hz=868e6, qos_path_loss_exp=2.2)
    distance = 500.0
    expected = (channel.wavelength_m / (4.0 * math.pi * distance)) ** channel.qos_path_loss_exp
    assert math.isclose(channel.qos_path_gain(distance), expected, rel_tol=1e-12)


def test_connection_probability_uses_expected_formula():
    channel = Channel(frequency_hz=868e6)
    distance = 1000.0
    sf = 7
    tx_power_dBm = 14.0
    channel.last_noise_dBm = -120.0
    gain = channel.qos_path_gain(distance)
    noise_lin = 10 ** (channel.last_noise_dBm / 10.0)
    snr_lin = 10 ** (channel.SNR_THRESHOLDS[sf] / 10.0)
    tx_lin = 10 ** (tx_power_dBm / 10.0)
    expected = math.exp(-noise_lin * snr_lin / (tx_lin * gain))
    assert math.isclose(
        channel.connection_probability(distance, sf, tx_power_dBm),
        expected,
        rel_tol=1e-12,
    )


def test_capture_probability_matches_reference():
    channel = Channel()
    nu_j = 0.75
    delta = 6.0
    expected = math.exp(-2.0 * nu_j) + (2.0 / (delta + 1.0)) * nu_j * math.exp(-2.0 * nu_j)
    assert math.isclose(channel.capture_probability(nu_j, delta), expected, rel_tol=1e-12)


@pytest.mark.parametrize("distance", [0.0, -10.0])
def test_qos_path_gain_invalid_distance(distance):
    channel = Channel()
    with pytest.raises(ValueError):
        channel.qos_path_gain(distance)


def test_capture_probability_negative_delta():
    channel = Channel()
    with pytest.raises(ValueError):
        channel.capture_probability(0.5, -1.0)
