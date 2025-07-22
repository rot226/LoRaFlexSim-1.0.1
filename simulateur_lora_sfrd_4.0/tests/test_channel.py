import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
import pytest  # noqa: E402
import random


def test_environment_preset_values():
    ch = Channel(environment="rural")
    assert ch.path_loss_exp == 2.0
    assert ch.shadowing_std == 2.0


def test_invalid_environment():
    with pytest.raises(ValueError):
        Channel(environment="unknown")


def test_new_environment_presets():
    dense = Channel(environment="urban_dense")
    indoor = Channel(environment="indoor")
    assert dense.path_loss_exp >= Channel.ENV_PRESETS["urban_dense"][0]
    assert indoor.path_loss_exp >= Channel.ENV_PRESETS["indoor"][0]


def test_region_preset_single_channel():
    ch = Channel(region="EU868", channel_index=1)
    assert ch.region == "EU868"
    assert ch.frequency_hz == Channel.REGION_CHANNELS["EU868"][1]


def test_region_channels_helper_returns_list():
    chans = Channel.region_channels("US915")
    assert len(chans) == len(Channel.REGION_CHANNELS["US915"])
    assert all(isinstance(c, Channel) for c in chans)


def test_spreading_gain_improves_snr():
    ch = Channel(shadowing_std=0)
    _, snr7 = ch.compute_rssi(14.0, 100.0, sf=7)
    _, snr12 = ch.compute_rssi(14.0, 100.0, sf=12)
    assert snr12 > snr7


def test_antenna_gains_increase_rssi():
    base = Channel(shadowing_std=0)
    gch = Channel(
        shadowing_std=0,
        tx_antenna_gain_dB=2.0,
        rx_antenna_gain_dB=3.0,
    )
    r1, _ = base.compute_rssi(14.0, 50.0)
    r2, _ = gch.compute_rssi(14.0, 50.0)
    assert r2 > r1


def test_offsets_and_system_loss():
    base = Channel(shadowing_std=0)
    mod = Channel(
        shadowing_std=0,
        system_loss_dB=2.0,
        rssi_offset_dB=3.0,
        snr_offset_dB=1.0,
    )
    r1, s1 = base.compute_rssi(14.0, 100.0, sf=7)
    r2, s2 = mod.compute_rssi(14.0, 100.0, sf=7)
    assert r2 == pytest.approx(r1 - 2.0 + 3.0)
    assert s2 == pytest.approx(s1 - 2.0 + 3.0 + 1.0)


def test_temperature_variation_changes_snr():
    random.seed(0)
    ch = Channel(shadowing_std=0, temperature_std_K=20.0)
    _, snr1 = ch.compute_rssi(14.0, 100.0)
    _, snr2 = ch.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_humidity_variation_changes_snr():
    random.seed(0)
    ch = Channel(
        shadowing_std=0,
        humidity_std_percent=10.0,
        humidity_noise_coeff_dB=1.0,
    )
    _, snr1 = ch.compute_rssi(14.0, 100.0)
    _, snr2 = ch.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_pa_non_linearity_channel():
    random.seed(0)
    ch = Channel(shadowing_std=0, pa_non_linearity_std_dB=1.0)
    r1, _ = ch.compute_rssi(14.0, 100.0)
    r2, _ = ch.compute_rssi(14.0, 100.0)
    assert r1 != r2


def test_phase_noise_channel():
    random.seed(0)
    ch = Channel(shadowing_std=0, phase_noise_std_dB=2.0)
    _, snr1 = ch.compute_rssi(14.0, 100.0)
    _, snr2 = ch.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_band_interference_degrades_snr():
    random.seed(0)
    base = Channel(shadowing_std=0)
    jam = Channel(
        shadowing_std=0,
        band_interference=[(base.frequency_hz, base.bandwidth, 10.0)],
    )
    _, snr_base = base.compute_rssi(14.0, 100.0)
    _, snr_jam = jam.compute_rssi(14.0, 100.0)
    assert snr_jam < snr_base


def test_fine_fading_variability():
    random.seed(0)
    ch = Channel(shadowing_std=0, fine_fading_std=1.0)
    r1, _ = ch.compute_rssi(14.0, 100.0)
    r2, _ = ch.compute_rssi(14.0, 100.0)
    assert r1 != r2


def test_variable_noise_changes_rssi():
    random.seed(0)
    ch = Channel(shadowing_std=0, variable_noise_std=2.0)
    _, s1 = ch.compute_rssi(14.0, 100.0)
    _, s2 = ch.compute_rssi(14.0, 100.0)
    assert s1 != s2


def test_frontend_filter_reduces_rssi():
    base = Channel(shadowing_std=0)
    filt = Channel(
        shadowing_std=0,
        frontend_filter_order=2,
        frontend_filter_bw=100e3,
        frequency_offset_hz=40e3,
    )
    r_base, _ = base.compute_rssi(14.0, 100.0)
    r_filt, _ = filt.compute_rssi(14.0, 100.0)
    assert r_filt < r_base
