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


def test_invalid_fading_model():
    with pytest.raises(ValueError):
        Channel(fading_model="invalid")


def test_rayleigh_fading_computation():
    ch = Channel(fading_model="rayleigh", fast_fading_std=1.0, shadowing_std=0)
    rssi, snr = ch.compute_rssi(14.0, 10.0)
    assert isinstance(rssi, float)
    assert isinstance(snr, float)
