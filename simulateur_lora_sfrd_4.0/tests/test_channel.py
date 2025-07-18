import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
import pytest  # noqa: E402


def test_environment_preset_values():
    ch = Channel(environment="rural")
    assert ch.path_loss_exp == 2.0
    assert ch.shadowing_std == 2.0


def test_invalid_environment():
    with pytest.raises(ValueError):
        Channel(environment="unknown")


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
