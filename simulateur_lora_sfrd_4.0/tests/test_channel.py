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
