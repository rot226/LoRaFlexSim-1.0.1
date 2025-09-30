import pytest

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.energy_profiles import EnergyProfile
from loraflexsim.launcher.node import Node


@pytest.fixture()
def energy_profile() -> EnergyProfile:
    return EnergyProfile(
        voltage_v=3.0,
        sleep_current_a=1e-3,
        rx_current_a=2e-3,
        listen_current_a=0.0,
        process_current_a=1.5e-3,
        tx_current_map_a={14.0: 5e-3},
    )


def make_node(profile: EnergyProfile) -> Node:
    ch = Channel()
    return Node(0, 0.0, 0.0, sf=7, tx_power=14.0, channel=ch, energy_profile=profile)


def test_sleep_energy_enforced(energy_profile: EnergyProfile) -> None:
    node = make_node(energy_profile)
    node.add_energy(0.0, "sleep", duration_s=2.0)
    expected = energy_profile.energy_for("sleep", 2.0)
    assert node.energy_sleep == pytest.approx(expected)
    assert node.energy_consumed == pytest.approx(expected)


def test_rx_energy_enforced(energy_profile: EnergyProfile) -> None:
    node = make_node(energy_profile)
    node.add_energy(1.0, "rx", duration_s=1.5)
    expected = energy_profile.energy_for("rx", 1.5)
    assert node.energy_rx == pytest.approx(expected)
    assert node.energy_consumed == pytest.approx(expected)


def test_tx_energy_enforced(energy_profile: EnergyProfile) -> None:
    profile = energy_profile
    profile = EnergyProfile(
        voltage_v=3.0,
        sleep_current_a=profile.sleep_current_a,
        rx_current_a=profile.rx_current_a,
        listen_current_a=profile.listen_current_a,
        process_current_a=profile.process_current_a,
        ramp_up_s=0.001,
        ramp_down_s=0.001,
        tx_current_map_a=profile.tx_current_map_a,
    )
    node = make_node(profile)
    node.add_energy(0.0, "tx", duration_s=0.5)
    tx = profile.energy_for("tx", 0.5, power_dBm=14.0)
    ramp = profile.energy_for("ramp", profile.ramp_up_s + profile.ramp_down_s, power_dBm=14.0)
    assert node.energy_tx == pytest.approx(tx)
    assert node.energy_ramp == pytest.approx(ramp)
    assert node.energy_consumed == pytest.approx(tx + ramp)
