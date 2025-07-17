import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
import pytest


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
