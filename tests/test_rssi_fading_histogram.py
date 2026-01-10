import numpy as np

from loraflexsim.launcher.channel import Channel


def _sample_rssi(channel: Channel, tx_power: float, distance: float, samples: int = 600):
    values = []
    for _ in range(samples):
        rssi, _snr = channel.compute_rssi(tx_power, distance)
        values.append(rssi)
    return np.array(values)


def test_rayleigh_fading_produces_variation():
    rng = np.random.default_rng(1234)
    channel = Channel(
        rng=rng,
        phy_model="none",
        shadowing_std=0.0,
        fast_fading_std=1.0,
        multipath_taps=1,
        time_variation_std=0.0,
        tx_power_std=0.0,
        noise_floor_std=0.0,
    )
    tx_power_dBm = 14.0
    distance = 100.0

    # exercise compute_rssi -> _rayleigh_fading_db for fast fading
    rssi_samples = _sample_rssi(channel, tx_power_dBm, distance)

    assert rssi_samples.std() > 0.1
    assert np.unique(rssi_samples).size > 1


def test_shadowing_log_normal_produces_variation():
    rng = np.random.default_rng(1234)
    channel = Channel(
        rng=rng,
        phy_model="none",
        shadowing_std=3.0,
        fast_fading_std=0.0,
        multipath_taps=1,
        time_variation_std=0.0,
        tx_power_std=0.0,
        noise_floor_std=0.0,
    )
    tx_power_dBm = 14.0
    distance = 100.0

    # exercise compute_rssi -> shadowing_std log-normal term
    rssi_samples = _sample_rssi(channel, tx_power_dBm, distance)

    assert rssi_samples.std() > 0.1
    assert np.unique(rssi_samples).size > 1
