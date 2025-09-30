import pytest

from loraflexsim.launcher.energy_profiles import EnergyProfile
from loraflexsim.launcher.node import Node
from loraflexsim.launcher.channel import Channel


def test_cumulative_energy_matches_flora_model():
    profile = EnergyProfile(
        voltage_v=1.0,
        startup_current_a=2.0,
        startup_time_s=1.0,
        preamble_current_a=3.0,
        preamble_time_s=1.0,
        ramp_up_s=1.0,
        ramp_down_s=1.0,
        tx_current_map_a={14.0: 4.0},
    )
    ch = Channel()
    node = Node(0, 0.0, 0.0, sf=7, tx_power=14.0, channel=ch, energy_profile=profile)
    airtime = 1.0
    tx_energy = profile.get_tx_current(14.0) * profile.voltage_v * airtime
    node.add_energy(tx_energy, "tx")
    expected = (
        tx_energy
        + profile.get_tx_current(14.0) * profile.voltage_v * (profile.ramp_up_s + profile.ramp_down_s)
        + profile.startup_current_a * profile.voltage_v * profile.startup_time_s
        + profile.preamble_current_a * profile.voltage_v * profile.preamble_time_s
    )
    assert node.energy_consumed == pytest.approx(expected)
    assert node.energy_startup == pytest.approx(
        profile.startup_current_a * profile.voltage_v * profile.startup_time_s
    )
    assert node.energy_preamble == pytest.approx(
        profile.preamble_current_a * profile.voltage_v * profile.preamble_time_s
    )
    assert node.energy_tx == pytest.approx(tx_energy)
    assert node.energy_ramp == pytest.approx(
        profile.get_tx_current(14.0)
        * profile.voltage_v
        * (profile.ramp_up_s + profile.ramp_down_s)
    )
    breakdown = node.get_energy_breakdown()
    assert breakdown["tx"] == pytest.approx(node.energy_tx)
    assert breakdown["ramp"] == pytest.approx(node.energy_ramp)
    assert breakdown["startup"] == pytest.approx(node.energy_startup)
    assert node.energy.total() == pytest.approx(node.energy_consumed)
