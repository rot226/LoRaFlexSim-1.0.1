import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.energy_profiles import EnergyProfile, register_profile, get_profile  # noqa: E402


def test_register_region_and_helper():
    Channel.register_region("TEST", [100.0, 200.0])
    chans = Channel.region_channels("TEST")
    assert len(chans) == 2
    assert chans[0].frequency_hz == 100.0
    assert chans[0].region == "TEST"


def test_energy_profile_registry_and_lookup():
    custom = EnergyProfile(voltage_v=3.0)
    register_profile("custom", custom)
    assert get_profile("custom") is custom
    node = Node(1, 0, 0, 7, 14.0, channel=Channel(), energy_profile="custom")
    assert node.profile is custom
